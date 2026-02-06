import torch
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.random_utility import SeedingScheme, RandomGenerator
from lm_wm_tools.watermarks.utils import resolve_top_k_from_sampling_params

class ChiTwoWatermark(WatermarkProtocol):
    """Chi-Squared watermark implementation for vLLM logits processor."""

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

        # Scheme parameters
        self.sigma = epsilon

        # Sampling parameters
        self.temperature = sampling_parameters.temperature
        self.vocab_size = vocab_size
        self.rng_device = rng_device
        self.top_k = resolve_top_k_from_sampling_params(sampling_parameters, vocab_size)

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

    def get_name(self):
        return f"Chi2/{self.seeding_scheme.value}/k_{self.context_size}/sigma_{self.sigma:<.2f}/topk_{self.top_k}/seed_{self.seed}"

    def _find_mu(self, p: torch.Tensor, G: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
        """
        Args:
            p: (B, N) Tensor of probabilities/weights
            G: (B, N) Tensor of values
            tol: Tolerance for numerical stability
        Returns:
            mu: (B,) Tensor of solutions
        """
        B, N = p.shape
        sigma = self.sigma

        tau = -1.0 - sigma * G  # Shape (B, N)

        tau_s, indices = torch.sort(tau, dim=1)
        p_s = torch.gather(p, 1, indices)
        G_s = torch.gather(G, 1, indices)

        P_cum = torch.cumsum(p_s, dim=1)
        

        term = p_s * (1.0 + sigma * G_s)
        A_cum = torch.cumsum(term, dim=1)

        U_candidates = (1.0 - A_cum) / (P_cum + 1e-18)

        cond_positive = P_cum > 1e-18
        cond_left = U_candidates >= (tau_s - tol)

        infinity = torch.full((B, 1), float('inf'), device=p.device, dtype=p.dtype)
        tau_next = torch.cat([tau_s[:, 1:], infinity], dim=1)
        cond_right = U_candidates <= (tau_next + tol)

        is_valid = cond_positive & cond_left & cond_right

        first_valid_idx = torch.argmax(is_valid.long(), dim=1) # Shape (B,)
        row_has_solution = is_valid.any(dim=1)

        final_idx = torch.where(
            row_has_solution, 
            first_valid_idx, 
            torch.tensor(N - 1, device=p.device)
        )

        U_selected = torch.gather(U_candidates, 1, final_idx.unsqueeze(1)).squeeze(1)

        return U_selected / sigma

    def _find_mu_bernoulli(self, p, G, tol=1e-12) -> float:
        sigma = float(self.sigma)

        # G is {0,1}, p sums to 1

        mask1 = G > 0.5

        P1 = p[mask1].sum()
        P0 = 1.0 - P1

        # candidates
        mu1 = -P1  # no truncation
        if P1 <= 1.0 / sigma + tol:  # U_all > -1 - tol
            return float(mu1)

        # otherwise: truncate the G=0 block
        if P1 <= tol:  # guard against divide-by-zero
            return float(mu1)  # same as P1==0 case
        mu0 = -1.0 - P0 / (sigma * P1)
        return float(mu0)

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

            scores = self.random_generator.sample(
                self.seed, context_hash, topk_indices
            ).squeeze(0)

            if (
                self.distribution_name == "binomial"
                and self.distribution_parameters.get("total_count", 0) == 1
            ):
                # Use the faster Bernoulli mu finder
                mu = self._find_mu_bernoulli(probs, scores)
            else:
                mu = self._find_mu(probs.unsqueeze(0), scores.unsqueeze(0)).item()

            min_value = torch.tensor(1e-30, device=logits.device, dtype=scores.dtype)
            adjusted_scores = torch.log(
                torch.maximum(1 + self.sigma * (scores + mu), min_value)
            )
            logits = logits.clone()
            logits[topk_indices] = logits[topk_indices] + adjusted_scores
        return logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:

        # Only consider top-k
        k = min(self.top_k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

        V = topk_probs.numel()
        n_mc = 128
        device = probs.device

        scores = self.random_generator.random_sample(V, n_mc, device)  # shape [n_mc, V]

        mu = self._find_mu(
            topk_probs.unsqueeze(0).expand(n_mc, -1), scores, tol=1e-12
        ).view(-1, 1)  # shape [n_mc, 1]

        adjusted_scores = torch.log(
            torch.maximum(
                1 + self.sigma * (scores + mu),
                torch.tensor(1e-30, device=device, dtype=scores.dtype),
            )
        )

        topk_logits = torch.log(topk_probs + 1e-30) * self.temperature
        topk_logits = topk_logits.unsqueeze(0).expand(n_mc, -1)  + adjusted_scores  # shape [n_mc, V]
        expected_probs = torch.softmax(topk_logits / self.temperature, dim=-1).mean(dim=0)

        # Expand back to full vocab size
        expanded_expected_probs = torch.zeros_like(probs)
        expanded_expected_probs[topk_indices] = expected_probs
        
        return expanded_expected_probs

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
