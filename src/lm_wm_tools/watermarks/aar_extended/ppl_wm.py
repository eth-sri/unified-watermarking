import torch
from vllm import SamplingParams
from typing import Dict, Tuple

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.random_utility import SeedingScheme, RandomGenerator
from lm_wm_tools.watermarks.utils import resolve_top_k_from_sampling_params


class PPLWatermark(WatermarkProtocol):
    """PPL constrained watermark implementation for vLLM logits processor."""

    def __init__(
        self,
        epsilon: float,
        vocab_size: int,
        rng_device: str,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        distribution_name: str = "binomial",
        distribution_parameters: dict = {"total_count": 30, "probs": 0.5},
        **kwargs,
    ) -> None:
        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)

        assert context_size >= 1, "context_size must be at least 1"

        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        # Scheme Parameters
        self.epsilon = epsilon

        # Sampling Parameters
        self.temperature = sampling_parameters.temperature
        self.vocab_size = vocab_size
        self.rng_device = rng_device
        self.top_k = resolve_top_k_from_sampling_params(
            sampling_parameters, vocab_size
        )

        # Initialize the seeding scheme
        seeding_scheme.initialize(vocab_size, seed, rng_device)
        self.seeding_scheme = seeding_scheme
        self.context_size = context_size

        # Intialize the RNG
        self.seed = seed
        self.random_generator = RandomGenerator(
            distribution_name, distribution_parameters
        )
        self.distribution_name = distribution_name
        self.distribution_parameters = distribution_parameters
        
        # Cache compiled bisection solvers keyed by device/dtype/shapes.
        self._bisection_solver_cache: dict = {}
        self.beta = None # Will be set during call

    def get_name(self) -> str:
        return f"PPL/{self.distribution_name}/{self.seeding_scheme.value}/k_{self.context_size}/epsilon_{self.epsilon}/seed_{self.seed}"

    @torch.no_grad()
    def find_beta_for_logp_constraint_new(
        self,
        p: torch.Tensor,
        n_mc: int,
        device: torch.device,
        max_iter: int = 60,  # bisection iterations
    ) -> Tuple[float, Dict]:
        """
        Solve for beta so that E_G[ log p_{i*(G; beta)} ] matches the target:
        target = sum_i p_i log p_i  (equality), or within a band [target-eps, target+eps].

        Args:
        p: 1D prob vector on device, length V, must be strictly positive (or will be clamped).
        n_mc: Monte Carlo sample size.
        device: torch.device.
        eps: relaxation band size; eps=0 enforces equality, eps>0 allows band feasibility.
        Returns:
        beta (float), and a dict of diagnostics.
        """
        V = p.numel()
        p = p.to(device)
        eps = self.epsilon
        sample_G = self.random_generator.random_sample

        # Ensure numerically safe logs, keep consistency for target computation
        p_safe = p.clamp_min(1e-12)
        logp = p_safe.log()
        target = (p_safe * logp).sum()

        # Fix MC draws for stability across iterations
        G = sample_G(V, n_mc, device)  # shape [n_mc, V]

        beta = self._get_bisection_solver(logp, G, max_iter)(
            torch.tensor(0.0, device=device, dtype=logp.dtype),
            torch.tensor(1e2, device=device, dtype=logp.dtype),
            logp,
            G,
            target,
            torch.tensor(eps, device=device, dtype=logp.dtype),
        )

        return beta.item(), {}


    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> torch.Tensor:
        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device), self.context_size
        )
        if context_hash is not None:

            k = min(self.top_k, logits.shape[-1])
            topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
            with torch.no_grad():
                probs = torch.softmax(topk_logits / self.temperature, dim=-1)

            beta, _ = self.find_beta_for_logp_constraint_new(
                probs,
                n_mc=128,
                device=logits.device,
            )
            self.beta = beta
            
            scores = self.random_generator.sample(self.seed, context_hash, topk_indices)
            scores = scores + beta * torch.log(probs.clamp_min(1e-12))
            argmax_idx = torch.argmax(scores).item()
            argmax_idx = topk_indices[argmax_idx].item()

            # Create a dirac distribution at the argmax index
            logits = torch.full_like(logits, -100.0)
            logits[argmax_idx] = 0.0

        return logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Given a prob vector p [V], return the expected probs under the watermarking
        selection policy.
        """       

        if self.beta is None:
            return probs

        # Only consider top-k
        k = min(self.top_k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        
        V = topk_probs.numel()
        n_mc = 128
        device = probs.device
        
        scores = self.random_generator.random_sample(V, n_mc, device)  # shape [n_mc, V]
        scores = scores + self.beta * torch.log(topk_probs.clamp_min(1e-12))
        argmax_idxs = torch.argmax(scores, dim=1)  # [n_mc]
        counts = torch.bincount(argmax_idxs, minlength=V).float()  # [V]
        expected_topk_probs = counts / n_mc  # [V]

        # Reconstruct full prob vector
        expected_probs = torch.zeros_like(probs)
        expected_probs[topk_indices] = expected_topk_probs
        
        return expected_probs

    def _get_bisection_solver(
        self, logp: torch.Tensor, G: torch.Tensor, max_iter: int
    ):
        """
        Lazily compile and cache a bisection solver that takes logp/G as inputs.
        """

        cache_key = (
            logp.device,
            logp.dtype,
            logp.shape,
            G.shape,
            max_iter,
        )
        if cache_key in self._bisection_solver_cache:
            return self._bisection_solver_cache[cache_key]

        compile_opts = {"mode": "default", "fullgraph": True}

        @torch.no_grad()
        def bisection_logic(
            beta_low: torch.Tensor,
            beta_high: torch.Tensor,
            logp_input: torch.Tensor,
            G_input: torch.Tensor,
            target_input: torch.Tensor,
            eps_input: torch.Tensor,
        ) -> torch.Tensor:
            # Runs a fixed number of iterations to keep the graph stable.
            for _ in range(max_iter):
                mid = (beta_low + beta_high) / 2.0
                s = G_input + mid * logp_input  # broadcast [n_mc, V]
                idx = s.argmax(dim=1)  # [n_mc]
                val = logp_input[idx].mean() - target_input + eps_input
                beta_high = torch.where(val > 0, mid, beta_high)
                beta_low = torch.where(val < 0, mid, beta_low)
            return (beta_low + beta_high) / 2.0

        compiled_solver = torch.compile(
            bisection_logic, **compile_opts, dynamic=False
        )
        self._bisection_solver_cache[cache_key] = compiled_solver
        return compiled_solver

    def detect(self, tokens: list[int] | torch.Tensor) -> float:
        scores = []

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.int32)

        tokens = tokens.to(self.rng_device)

        context, tokens = self.seeding_scheme.hash_context(tokens, self.context_size)
        pairs = torch.stack((context, tokens), dim=1)  # shape: [N, 2]
        unique_pairs = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=False, return_counts=False
        )
        offsets = torch.hash_tensor(unique_pairs, dim=1)

        scores = self.random_generator._sample(self.seed, offsets)

        out = self.random_generator.statistical_test(scores)

        return out
