from lm_wm_tools.utils.triton_code import stateless_uniform
from enum import Enum
import torch
from typing import Optional
from math import sqrt
import numpy as np

from scipy.stats import (
    kstest,
    binomtest,
    norm
)


class SeedingScheme(Enum):
    SUMHASH = "sumhash"
    MINHASH = "minhash"

    def initialize(self, vocab_size: int, seed: int, device: torch.device):
        # Random vocabulary permutation for the hashing scheme
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.permutation = torch.randperm(
            vocab_size, generator=g, device=device
        )

    def hash_last_context(self, inputs: torch.Tensor, context_size: int) -> Optional[int]:
        """Compute the hash of the latest context in the inputs. Only works with 1-D tensors"""

        assert inputs.dim() == 1, "Inputs must be a 1-D tensor"
        assert self.permutation is not None, "Permutation not initialized. Call initialize() first."

        if inputs.size(0) < context_size:
            return None

        if self.value == "sumhash":
            return torch.sum(
                self.permutation[inputs[-context_size :]]
            ).item() % (2**32 - 1)
        elif self.value == "minhash":
            return torch.min(
                self.permutation[inputs[-context_size :]]
            ).item() % (2**32 - 1)


    def hash_context(self, inputs: torch.Tensor, context_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the hash of all the contexts.
        
        Returns the context hashes and the corresponding inputs
        """

        assert inputs.shape[-1] >= context_size, "Input length must be at least the context size"
        assert self.permutation is not None, "Permutation not initialized. Call initialize() first."

        unfolded_inputs = inputs.unfold(dimension=-1, size=context_size, step=1)

        if self.value == "sumhash":
            hash_tensor = torch.sum(
                self.permutation[unfolded_inputs]
            , dim=-1) % (2**32 - 1) 
        elif self.value == "minhash":
            hash_tensor = torch.min(
                self.permutation[unfolded_inputs]
            , dim=-1) % (2**32 - 1) 
        
        return hash_tensor[:-1], inputs[context_size:]
    

class Binomial(torch.distributions.Binomial):

    def __init__(self, **kwargs):
       super().__init__(**kwargs)

    def icdf_bernoulli(self, uniforms: torch.Tensor) -> torch.Tensor:
        """This is simply the Bernoulli icdf. To get a Binomial, call it total_counts times and sum."""
        return (uniforms < self.probs).to(torch.int32)
    

DISTRIBUTION_MAP = {
    "normal": torch.distributions.Normal,
    "uniform": torch.distributions.Uniform,
    "binomial": Binomial,
    "gumbel": torch.distributions.Gumbel,
    "lognormal": torch.distributions.LogNormal,
}


class TestResult:
    def __init__(self, statistic: float, pvalue: float):
        self.statistic = statistic
        self.pvalue = pvalue

class RandomGenerator():

    def __init__(self, distribution_name: str, distribution_parameters: dict[str, float]):

        self.distribution_name = distribution_name
        self.distribution_parameters = distribution_parameters
        self.distribution = DISTRIBUTION_MAP[distribution_name](**distribution_parameters)

    def _get_scipy_args(self):
        """Maps PyTorch distribution parameters to Scipy arguments."""
        params = self.distribution_parameters
        
        if self.distribution_name == "uniform":
            # PyTorch: low, high -> Scipy: loc=low, scale=high-low
            return (params["low"], params["high"] - params["low"])
            
        elif self.distribution_name == "gumbel":
            # PyTorch: loc, scale -> Scipy: loc, scale
            # We explicitly fetch keys to ensure order, rather than relying on .values()
            return (params["loc"], params["scale"])
            
        elif self.distribution_name == "normal":
            return (params["loc"], params["scale"])
            
        elif self.distribution_name == "lognormal":
             return (params["loc"], params["scale"])
             
        return tuple(params.values())

    def _sample(self, seed: int, offsets: torch.Tensor) -> torch.Tensor:

        if self.distribution_name == "binomial":
            seeds = [seed + i for i in range(self.distribution_parameters["total_count"])]
            distribution = torch.stack([self.distribution.icdf_bernoulli(stateless_uniform(seed=s, offsets=offsets)) for s in seeds]).sum(dim=0)
        else:
            uniform = stateless_uniform(seed=seed, offsets=offsets)
            distribution = self.distribution.icdf(uniform)

        return distribution

    def sample(self, seed: int, context_hashes: torch.Tensor | int, tokens: torch.Tensor) -> torch.Tensor:

        if isinstance(context_hashes, int):
            context_hashes = torch.tensor([context_hashes], device=tokens.device, dtype=torch.int32)

        num_ctx = context_hashes.size(0)
        num_tok = tokens.size(0)

        # Create all (context_hash, token) pairs as a grid
        ctx_grid = context_hashes.view(num_ctx, 1).expand(num_ctx, num_tok)   # (C, T)
        tok_grid = tokens.view(1, num_tok).expand(num_ctx, num_tok)          # (C, T)

        # Stack into pairs and hash along the last dimension
        pairs = torch.stack((ctx_grid, tok_grid), dim=-1)  # (C, T, 2)
        offsets = torch.hash_tensor(pairs, dim=-1)         # (C, T)

        offsets_flat = offsets.reshape(-1)                 # (C * T,)
        samples_flat = self._sample(seed, offsets_flat)

        samples = samples_flat.reshape(num_ctx, num_tok)

        return samples
        
    def random_sample(self, V: int, n_mc: int, device: torch.device) -> torch.Tensor:
        """Random sampling for Monte-Carlo estimation. Do not use for watermarking as it is fully random."""
        samples = self.distribution.sample((n_mc, V)).to(device=device)
        return samples
    
    def statistical_test(self, scores: torch.Tensor | list[int]):

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        scipy_args = self._get_scipy_args()

        if self.distribution_name == "gumbel":
            result = kstest(scores, "gumbel_r", args=scipy_args, alternative="less")
        elif self.distribution_name == "uniform":
            result = kstest(scores, "uniform", args=scipy_args, alternative="less")
        elif self.distribution_name == "binomial":
            # We use a binomial test on the sum
            total_scores = sum(scores)
            n_values = len(scores)

            n_binomial = self.distribution_parameters["total_count"]
            p_binomial = self.distribution_parameters["probs"]

            result = binomtest(total_scores, n=n_values * n_binomial, p=p_binomial, alternative="greater")

        elif self.distribution_name == "normal":
            loc, scale = scipy_args
            # Standard Z-test
            z_score = (np.mean(scores) - loc) / (scale / sqrt(len(scores)))
            p_value = float(norm.sf(z_score))
            z_score = float(z_score)
            result = TestResult(statistic=z_score, pvalue=p_value)

        elif self.distribution_name == "lognormal":
            loc = self.distribution_parameters["loc"]
            scale = self.distribution_parameters["scale"]
            
            log_scores = np.log(scores)

            z_score = (np.mean(log_scores) - loc) / (scale / sqrt(len(scores)))
            
            p_value = float(norm.sf(z_score))
            z_score = float(z_score)
            result = TestResult(statistic=z_score, pvalue=p_value)

        out = {
            "statistic": result.statistic,
            "pvalue": result.pvalue
        }

        return out


