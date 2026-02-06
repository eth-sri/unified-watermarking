from vllm import SamplingParams


def resolve_top_k_from_sampling_params(
    sampling_parameters: SamplingParams, vocab_size: int
) -> int:
    """
    Determine the effective top-k to use for watermarking from vLLM sampling parameters.
    vLLM stores "no top-k" as -1, so default to the full vocabulary in that case.
    """
    sp_top_k = getattr(sampling_parameters, "top_k", None)
    if sp_top_k is None or sp_top_k == -1:
        return vocab_size
    if sp_top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return min(sp_top_k, vocab_size)
