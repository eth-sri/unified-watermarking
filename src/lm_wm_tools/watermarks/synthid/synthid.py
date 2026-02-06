"""SynthID watermarking utilities and vLLM logits processor."""

import torch
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.utils import resolve_top_k_from_sampling_params
from lm_wm_tools.watermarks.random_utility import SeedingScheme, RandomGenerator
from lm_wm_tools.utils.triton_code import stateless_uniform


def update_scores(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
) -> torch.FloatTensor:
  """Updates scores using the g values.

  We assume that the scores are in the log space.
  Args:
    scores: Scores (batch_size, vocab_size).
    g_values: G values (batch_size, vocab_size, depth).

  Returns:
    Updated scores (batch_size, vocab_size).
  """
  _, _, depth = g_values.shape
  device = scores.device

  probs = torch.softmax(scores, dim=1)

  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

  log_probs = torch.log(probs)
  log_probs = torch.where(
      torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
  )
  return log_probs


def update_scores_distortionary(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
    num_leaves: int,
) -> torch.FloatTensor:
  """Update scores using the g values for distortionary tournament watermarking.

  We assume that the scores are in the log space.
  Args:
    scores: Scores (batch_size, vocab_size).
    g_values: G values (batch_size, vocab_size, depth).
    num_leaves: Number of leaves per node in the tournament tree.

  Returns:
    Updated scores (batch_size, vocab_size).
  """
  _, _, depth = g_values.shape
  device = scores.device

  probs = torch.softmax(scores, dim=1)

  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    coeff_not_in_g = (1 - g_mass_at_depth) ** (num_leaves - 1)
    coeff_in_g = (1 - (1 - g_mass_at_depth) ** (num_leaves)) / g_mass_at_depth
    coeffs = torch.where(
        torch.logical_and(g_values_at_depth == 1, probs > 0),
        coeff_in_g,
        coeff_not_in_g,
    )
    probs = probs * coeffs

  log_probs = torch.log(probs)
  log_probs = torch.where(
      torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
  )
  return log_probs


class SynthIDWatermark(WatermarkProtocol):
    """SynthID watermarking implementation for vLLM logits processors.

    Simplified version without n-gram caching.
    """

    def __init__(
        self,
        epsilon: int,
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

        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        assert distribution_name == "binomial", (
            "Only binomial distribution is supported for SynthID watermarking."
        )

        # If epsilon is not an integer (or cannot be cast to int without loss), raise error
        if not isinstance(epsilon, int):
            self._num_leaves = int(epsilon)
            if abs(self._num_leaves - epsilon) > 1e-6:
                raise ValueError("Epsilon must be an integer for SynthID watermarking.")

        self.depth = distribution_parameters["total_count"]

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

    def get_name(self) -> str:
        return f"SynthID/k_{self.context_size}/depth_{self.depth}/seed_{self.seed}"

    def get_gvalues(self, hash: int, tokens: torch.Tensor) -> torch.Tensor:
        """Get the G values for all tokens given a context hash."""

        context_hashes = torch.tensor([hash], device=tokens.device, dtype=torch.int32)

        num_ctx = context_hashes.size(0)
        num_tok = tokens.size(0)

        # Create all (context_hash, token) pairs as a grid
        ctx_grid = context_hashes.view(num_ctx, 1).expand(num_ctx, num_tok)  # (C, T)
        tok_grid = tokens.view(1, num_tok).expand(num_ctx, num_tok)  # (C, T)

        # Stack into pairs and hash along the last dimension
        pairs = torch.stack((ctx_grid, tok_grid), dim=-1)  # (C, T, 2)
        offsets = torch.hash_tensor(pairs, dim=-1)  # (C, T)

        offsets = offsets.reshape(-1)  # (C * T,)

        seeds = [
            self.seed + i for i in range(self.distribution_parameters["total_count"])
        ]
        distribution = torch.stack(
            [
                self.random_generator.distribution.icdf_bernoulli(
                    stateless_uniform(seed=s, offsets=offsets)
                )
                for s in seeds
            ]
        ).squeeze(1)
        return distribution.T  # shape: (vocab_size, m)

    @torch.no_grad()
    def __call__(
        self,
        output_ids: list[int],
        logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Applies the SynthID watermark to the provided logits."""
        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device), self.context_size
        )
        if context_hash is None:
            return logits

        logits_scaled = logits / self.temperature

        k = min(self.top_k, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits_scaled, k, dim=-1)
        g_values = self.get_gvalues(context_hash, topk_indices)

        # Update scores function expect batch dimension
        g_values = g_values.unsqueeze(0)
        topk_logits = topk_logits.unsqueeze(0)

        if self._num_leaves == 2:
            updated_log_probs = update_scores(topk_logits, g_values)
        else:
            updated_log_probs = update_scores_distortionary(
                topk_logits, g_values, self._num_leaves
            )

        updated_log_probs = updated_log_probs.squeeze(0)

        original_log_probs = torch.log_softmax(topk_logits, dim=0)
        deltas = (updated_log_probs - original_log_probs).squeeze(0)

        logits_scaled = logits_scaled.clone()
        logits_scaled[topk_indices] = logits_scaled[topk_indices] + deltas

        return logits_scaled * self.temperature

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:

        # Only consider top-k
        k = min(self.top_k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

        n_mc = 128
        device = probs.device
        eps = 1e-30

        logits_scaled = torch.log(probs.clamp_min(eps))
        topk_logits_scaled = logits_scaled[topk_indices].unsqueeze(0).expand(n_mc, -1)
        topk_mass = topk_probs.sum()

        # Sample g-values from the same Bernoulli distribution used in __call__
        g_prob = self.distribution_parameters.get("probs", 0.5)
        g_values = torch.bernoulli(
            torch.full(
                (n_mc, topk_probs.numel(), self.depth),
                g_prob,
                device=device,
                dtype=probs.dtype,
            )
        )

        if self._num_leaves == 2:
            updated_log_probs = update_scores(topk_logits_scaled, g_values)
        else:
            updated_log_probs = update_scores_distortionary(
                topk_logits_scaled,
                g_values,
                self._num_leaves,
            )

        expected_topk_probs = topk_mass * updated_log_probs.exp().mean(dim=0)
        
        expanded_expected_probs = probs.clone()
        expanded_expected_probs[topk_indices] = expected_topk_probs

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
