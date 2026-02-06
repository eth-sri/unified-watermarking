import torch
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.random_utility import SeedingScheme, RandomGenerator
from lm_wm_tools.watermarks.utils import resolve_top_k_from_sampling_params



class PPLSingleFastWatermark(WatermarkProtocol):
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

        # Scheme parameters
        self.epsilon = epsilon

        # Sampling parameters
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

    def get_name(self) -> str:
        return f"PPLSingle/{self.seeding_scheme.value}/k_{self.context_size}/epsilon_{self.epsilon}/seed_{self.seed}"
    
    def convex_hull_algorithm(self, probs: torch.Tensor, scores: torch.Tensor, epsilon: float) -> torch.Tensor:
        probs = probs.unsqueeze(0)  # shape [1, V]
        scores = scores.unsqueeze(0)  # shape [1, V]
        best_q = convex_hull_algorithm_vectorized(probs, scores, epsilon)
        return best_q.squeeze(0)  # shape [V]

    @torch.no_grad()
    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor, # (vocab_size)
    ) -> torch.Tensor:
        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device), self.context_size
        )
        if context_hash is not None:

            logits_scaled = logits / self.temperature

            k = min(self.top_k, logits.shape[-1])
            topk_logits, topk_indices = torch.topk(logits_scaled, k, dim=-1)
            probs = torch.softmax(topk_logits, dim=-1)
            scores = self.random_generator.sample(self.seed, context_hash, topk_indices).squeeze(0)

            probs = self.convex_hull_algorithm(probs, scores, self.epsilon)

            updated_log_probs = torch.log(probs)
            updated_log_probs = torch.where(
                torch.isfinite(updated_log_probs), updated_log_probs, torch.tensor(-1e12, device=logits.device)
            )

            logits = torch.full_like(logits_scaled, -100.0)
            logits[topk_indices] = updated_log_probs * self.temperature

        return logits


    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Monte-Carlo estimate of the expected probabilities after watermarking.
        Batches over the MC dimension for speed.
        """

        k = min(self.top_k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

        V = topk_probs.numel()
        n_mc = 128
        device = probs.device

        scores = self.random_generator.random_sample(V, n_mc, device)

        mc_probs = convex_hull_algorithm_vectorized(
            topk_probs.unsqueeze(0).expand(n_mc, -1),  # shape [n_mc, V]
            scores,  # shape [n_mc, V]
            self.epsilon,
        )  # shape [n_mc, V]
        expected_topk_probs = mc_probs.mean(dim=0)

        expanded_expected_probs = torch.zeros_like(probs)
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

def convex_hull_algorithm_vectorized(probs: torch.Tensor, scores: torch.Tensor, epsilon: float) -> torch.Tensor:

    B, V = probs.shape
    device = probs.device

    probs_safe = probs.clamp(min=1e-30)
    c = -torch.log(probs_safe)
    
    H_p = -(probs_safe * torch.log(probs_safe)).sum(dim=1, keepdim=True)
    C = H_p + epsilon

    c_sorted, idx_sorted = torch.sort(c, dim=1)
    scores_sorted = torch.gather(scores, 1, idx_sorted)
    
    running_max, _ = torch.cummax(scores_sorted, dim=1)
    
    is_pareto = scores_sorted >= running_max

    range_indices = torch.arange(V, device=device).unsqueeze(0).expand(B, V)
    sort_keys = torch.where(is_pareto, range_indices, range_indices + V)
    
    _, compressed_order = torch.sort(sort_keys, dim=1)
    
    num_valid = is_pareto.sum(dim=1)
    V_eff = num_valid.max().item()
    
    active_indices = compressed_order[:, :V_eff]
    
    c_active = torch.gather(c_sorted, 1, active_indices)
    s_active = torch.gather(scores_sorted, 1, active_indices)
    
    valid_mask = torch.arange(V_eff, device=device).unsqueeze(0) < num_valid.unsqueeze(1)
    
    hull_stack = torch.zeros((B, V_eff), dtype=torch.long, device=device)
    stack_ptr = torch.zeros(B, dtype=torch.long, device=device)
    batch_range = torch.arange(B, device=device)

    for i in range(V_eff):
        curr_c = c_active[:, i]
        curr_s = s_active[:, i]
        
        is_real_point = valid_mask[:, i]

        while True:
            mask_size = stack_ptr >= 2
            
            active_check = mask_size & is_real_point
            if not active_check.any():
                break

            idx_A = hull_stack[batch_range, stack_ptr - 2]
            idx_B = hull_stack[batch_range, stack_ptr - 1]

            Ax = c_active[batch_range, idx_A]
            Ay = s_active[batch_range, idx_A]
            Bx = c_active[batch_range, idx_B]
            By = s_active[batch_range, idx_B]

            cross = (Bx - Ax) * (curr_s - By) - (By - Ay) * (curr_c - Bx)
            
            mask_turn = cross >= 0.0
            pop_mask = active_check & mask_turn

            if not pop_mask.any():
                break
                
            stack_ptr[pop_mask] -= 1

        hull_stack[batch_range, stack_ptr] = i
        stack_ptr = stack_ptr + is_real_point.long()

    col_idx = torch.arange(V_eff, device=device).unsqueeze(0).expand(B, V_eff)
    in_hull_mask = col_idx < stack_ptr.unsqueeze(1)
    
    hull_indices = hull_stack
    hull_costs = torch.gather(c_active, 1, hull_indices)
    
    max_val = torch.finfo(hull_costs.dtype).max
    hull_costs_padded = hull_costs.clone()
    hull_costs_padded[~in_hull_mask] = max_val

    target_idx = torch.searchsorted(hull_costs_padded, C)

    idx_j_stack = target_idx.squeeze(1).clamp(max=V_eff - 1)
    idx_i_stack = (idx_j_stack - 1).clamp(min=0)
    
    last_valid = (stack_ptr - 1).clamp(min=0)
    use_last = idx_j_stack > last_valid
    idx_j_stack = torch.where(use_last, last_valid, idx_j_stack)
    idx_i_stack = torch.where(use_last, last_valid, idx_i_stack)

    idx_active_i = hull_stack[batch_range, idx_i_stack]
    idx_active_j = hull_stack[batch_range, idx_j_stack]

    c_i = c_active[batch_range, idx_active_i]
    c_j = c_active[batch_range, idx_active_j]
    
    C_flat = C.squeeze(1)
    denom = c_j - c_i
    is_same = denom.abs() < 1e-20
    alpha = (c_j - C_flat) / (denom + 1e-30)
    alpha = torch.where(is_same, torch.ones_like(alpha), alpha)
    alpha = alpha.clamp(0.0, 1.0)

    sorted_idx_i = torch.gather(active_indices, 1, idx_active_i.unsqueeze(1)).squeeze(1)
    sorted_idx_j = torch.gather(active_indices, 1, idx_active_j.unsqueeze(1)).squeeze(1)

    orig_i = idx_sorted[batch_range, sorted_idx_i]
    orig_j = idx_sorted[batch_range, sorted_idx_j]

    best_q = torch.zeros_like(probs)
    best_q.scatter_add_(1, orig_i.unsqueeze(1), alpha.unsqueeze(1))
    best_q.scatter_add_(1, orig_j.unsqueeze(1), (1.0 - alpha).unsqueeze(1))

    return best_q