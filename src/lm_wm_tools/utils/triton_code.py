import torch
import triton
import triton.language as tl

# CUDA backend

@triton.jit
def _stateless_uniform_kernel(
    offsets_ptr,          # *int32/int64* offsets
    out_ptr,              # float32
    n_elements,           # int32/int64
    seed,                 # int32/64 scalar
    BLOCK: tl.constexpr,  # block size
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    idx = block_start + tl.arange(0, BLOCK)
    mask = idx < n_elements

    # Load offsets for this block
    offs = tl.load(offsets_ptr + idx, mask=mask)
    # tl.rand expects int32 offsets
    offs = offs.to(tl.int32)

    # Stateless RNG: depends only on (seed, offs)
    rnd = tl.rand(seed, offs)  # U(0,1) float32

    tl.store(out_ptr + idx, rnd, mask=mask)


def _stateless_uniform_cuda(offsets: torch.Tensor, seed: int, block_size: int = 1024) -> torch.Tensor:
    """
    Stateless Philox-based RNG using Triton's tl.rand. Pytorch has no stateless RNG.

    Args
    ----
    offsets : torch.Tensor
        CUDA tensor of integer offsets (any shape, any int dtype).
    seed : int
        Global seed (Python int or scalar convertible to int32/64).
    block_size : int
        Triton block size (tuning knob).

    Returns
    -------
    torch.Tensor
        Float32 tensor of same shape as `offsets`, values in [0, 1).
    """
    if not offsets.is_cuda:
        raise ValueError("offsets must be a CUDA tensor")

    if not offsets.is_contiguous():
        offsets = offsets.contiguous()

    orig_shape = offsets.shape
    flat_offsets = offsets.view(-1)

    if flat_offsets.dtype not in (torch.int32, torch.int64):
        flat_offsets = flat_offsets.to(torch.int64)

    n_elements = flat_offsets.numel()

    out_flat = torch.empty(n_elements, device=offsets.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)

    _stateless_uniform_kernel[grid](
        flat_offsets,
        out_flat,
        n_elements,
        seed,
        BLOCK=block_size,
    )

    return out_flat.view(orig_shape)


# CPU backend
@torch.jit.script
def _stateless_uniform_cpu(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Generates a tensor of stateless pseudo-random uniform variables in [0, 1).
    
    Args:
        offsets: A tensor of integer offsets (of any shape).
        seed: An integer seed.
    
    Returns:
        A tensor of the same shape as `offsets` with values in [0, 1).
    """

    u = offsets.to(torch.int32) + seed
    
    u ^= (u >> 16)
    u *= -2048144789
    u ^= (u >> 13)
    u *= -1028477387
    u ^= (u >> 16)
    
    return (u & 0x7FFFFF).float() * (1.0 / 8388608.0)

def stateless_uniform(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Stateless RNG similar to Triton's tl.rand(seed, offsets).

    Args
    ----
    offsets : torch.Tensor
        Integer tensor (any shape) whose values act as counters.
        Device: CPU or CUDA. Else will be cast to CPU.
    seed : int
        Scalar seed. Changing this changes the entire random field.

    Returns
    -------
    torch.Tensor
        float32 tensor in [0,1), same shape & device as `offsets`.

    """
    if offsets.device.type == "cuda":
        return _stateless_uniform_cuda(offsets, seed)
    else:

        if offsets.device.type != "cpu":
            device_type = offsets.device.type
            print(f"Warning: Using CPU backend for device type {device_type}")
            offsets = offsets.to("cpu")

            uniform = _stateless_uniform_cpu(offsets, seed).to(device_type)
            return uniform

        return _stateless_uniform_cpu(offsets, seed)