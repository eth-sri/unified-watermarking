import torch
from typing import Dict, List
import os
import json
from dataclasses import dataclass


@dataclass
class WmStats:
    kl_divergence: float = 0.0
    chi2_divergence: float = 0.0
    ppl_hard: float = 0.0
    ppl_soft: float = 0.0
    kl_divergence_soft: float = 0.0
    count: int = 0

    def compute_metrics(self, original_probs: torch.Tensor, modified_probs: torch.Tensor, expected_probs: torch.Tensor) -> None:

        # Clamb the probs and renormalize
        original_probs = original_probs.clamp_min(1e-12)
        original_probs = original_probs / original_probs.sum()  
        modified_probs = modified_probs.clamp_min(1e-12)
        modified_probs = modified_probs / modified_probs.sum()
        expected_probs = expected_probs.clamp_min(1e-12)
        expected_probs = expected_probs / expected_probs.sum()


        # Log probabilities for numerical stability
        modified_log_probs = modified_probs.log()
        expected_log_probs = expected_probs.log()
        original_log_probs = original_probs.log()

        # Avoid redundant computations
        ppl_target = (original_probs * original_log_probs).sum().item()

        self.kl_divergence += torch.nn.functional.kl_div(
            modified_log_probs, original_log_probs, reduction='batchmean', log_target=True
        ).item()
        self.chi2_divergence += torch.sum(
            (modified_probs - original_probs) ** 2 / (original_probs + 1e-12)
        ).item()
        self.ppl_hard += - torch.sum(
            modified_probs * original_log_probs
        ).item() + ppl_target
        self.ppl_soft += - torch.sum(
            expected_probs * original_log_probs
        ).item() + ppl_target

        top_k = 50
        original_log_probs_topk, original_topk_indices = torch.topk(original_log_probs, top_k, dim=-1)
        expected_log_probs_topk = expected_log_probs[original_topk_indices]

        # Add minimal value to avoid NaNs in KL when expected_log_probs_topk has zeros due to MC sampling
        n_mc = 128
        expected_log_probs_topk = expected_log_probs_topk.clamp_min(torch.log(torch.tensor(1.0 / n_mc, device=expected_log_probs_topk.device)))
        expected_log_probs_topk = expected_log_probs_topk - torch.logsumexp(expected_log_probs_topk, dim=-1, keepdim=True)

        self.kl_divergence_soft += torch.nn.functional.kl_div(
            expected_log_probs_topk, original_log_probs_topk, reduction='batchmean', log_target=True
        ).item()
        self.count += 1

    def to_dict(self) -> Dict:
        return {
            "kl_divergence": self.kl_divergence,
            "chi2_divergence": self.chi2_divergence,
            "ppl_hard": self.ppl_hard,
            "ppl_soft": self.ppl_soft,
            "kl_divergence_soft": self.kl_divergence_soft,
            "count": self.count,
        }

    def update(self, other: 'WmStats') -> None:
        self.kl_divergence += other.kl_divergence
        self.chi2_divergence += other.chi2_divergence
        self.ppl_hard += other.ppl_hard
        self.ppl_soft += other.ppl_soft
        self.kl_divergence_soft += other.kl_divergence_soft
        self.count += other.count

    def average(self) -> 'WmStats':
        if self.count == 0:
            return WmStats()
        return WmStats(
            kl_divergence=self.kl_divergence / self.count,
            chi2_divergence=self.chi2_divergence / self.count,
            ppl_hard=self.ppl_hard / self.count,
            ppl_soft=self.ppl_soft / self.count,
            kl_divergence_soft=self.kl_divergence_soft / self.count,
            count=self.count,
        )


class LogitsMetricWrapper:
    def __init__(self, watermark_processor, request_id: str, metrics_dir: str, temperature: float, top_k: int = -1):
        
        self.processor = watermark_processor
        self.filepath = get_metrics_filepath(request_id, metrics_dir)
        self.temperature = temperature
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        original_logits = logits.clone()
        modified_logits = self.processor(token_ids, logits)

        if original_logits is modified_logits:
            return modified_logits # No modification done by the processor, skip metrics calculation

        orginal_probs = torch.nn.functional.softmax(original_logits / self.temperature, dim=-1)
        modified_probs = torch.nn.functional.softmax(modified_logits / self.temperature, dim=-1)
        expected_probs = self.processor.get_expected_probs(orginal_probs)
        #expected_probs = modified_probs.clone()

        # Metric calculation 
        wm_stats = WmStats()
        wm_stats.compute_metrics(orginal_probs, modified_probs, expected_probs)
        data = wm_stats.to_dict()

        # Write to the specific file passed in the constructor
        with open(self.filepath, "a") as f:
            f.write(json.dumps(data) + "\n")

        return modified_logits


def get_metrics_filepath(request_id: str, metrics_dir: str) -> str:
    file_path = os.path.join(metrics_dir, f"{request_id}.jsonl")
    return file_path

def summarize_metrics(request_id: str, metrics_dir: str) -> Dict:
    """
    Summarizes the collected metrics for a given request_id.
    Returns average metrics.
    """
    
    filepath = get_metrics_filepath(request_id, metrics_dir)
    if not os.path.exists(filepath):
        return {}

    aggregated_stats = WmStats()
    for line in open(filepath, "r"):
        data = json.loads(line)
        wm_stats = WmStats(
            kl_divergence=data["kl_divergence"],
            chi2_divergence=data["chi2_divergence"],
            ppl_hard=data["ppl_hard"],
            ppl_soft=data["ppl_soft"],
            kl_divergence_soft=data["kl_divergence_soft"],
            count=data["count"],
        )
        aggregated_stats.update(wm_stats)   
    if aggregated_stats.count == 0:
        return {}
    avg_stats = aggregated_stats.average()
    return avg_stats.to_dict()