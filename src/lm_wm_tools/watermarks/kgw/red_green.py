import torch
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.random_utility import SeedingScheme, RandomGenerator
from lm_wm_tools.watermarks.utils import resolve_top_k_from_sampling_params


class RedGreenWatermark(WatermarkProtocol):
    """Red-Green watermark implementation for vLLM logits processor."""

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
        distribution_parameters: dict = {"total_count": 1, "probs": 0.5},
        **kwargs,
    ) -> None:
        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)
            
        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        # Scheme parameters
        self.delta = epsilon

        # Sampling parameters
        self.vocab_size = vocab_size
        self.rng_device = rng_device
        self.top_k = resolve_top_k_from_sampling_params(
            sampling_parameters, vocab_size
        )
        self.temperature = sampling_parameters.temperature

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

    def get_name(self) -> str:
        return f"KGW/{self.seeding_scheme.value}/k_{self.context_size}/delta_{self.delta:<.2f}/seed_{self.seed}"

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> torch.Tensor:
        context_hash = self.seeding_scheme.hash_last_context(torch.tensor(output_ids, device=self.rng_device), self.context_size)
        if context_hash is not None:

            k = min(self.top_k, logits.shape[-1])
            topk_logits, topk_indices = torch.topk(logits, k, dim=-1)

            scores = self.random_generator.sample(
                self.seed,
                context_hash,
                topk_indices,
            ).squeeze(0) * self.delta
            logits[topk_indices] = topk_logits + scores.to(logits.device)
        return logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Given a prob vector p [V], return the expected probs under the watermarking
        selection policy.
        """       

        # Only consider top-k
        k = min(self.top_k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        
        V = topk_probs.numel()
        n_mc = 128
        device = probs.device

        topk_logits = torch.log(topk_probs + 1e-30) * self.temperature

        scores = self.random_generator.random_sample(V, n_mc, device)  # shape [n_mc, V]
        scores = scores * self.delta

        topk_logits = topk_logits.unsqueeze(0).expand(n_mc, -1)  + scores  # shape [n_mc, V]
        expected_probs = torch.softmax(topk_logits / self.temperature, dim=-1).mean(dim=0)
        # Expand back to full vocab size
        expanded_expected_probs = torch.zeros_like(probs)
        expanded_expected_probs[topk_indices] = expected_probs

        return expanded_expected_probs

    def detect(self, tokens: list[int] | torch.Tensor) -> float:
        gumbel_scores = []

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.int32)

        tokens = tokens.to(self.rng_device)

        context, tokens = self.seeding_scheme.hash_context(tokens, self.context_size)
        pairs = torch.stack((context, tokens), dim=1)  # shape: [N, 2]
        unique_pairs = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=False, return_counts=False
        )
        offsets = torch.hash_tensor(unique_pairs, dim=1)

        gumbel_scores = self.random_generator._sample(self.seed, offsets)

        out = self.random_generator.statistical_test(gumbel_scores)

        return out
