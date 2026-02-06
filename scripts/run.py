#!/usr/bin/env python
"""Queue vLLM watermark sweeps with per-distribution epsilon defaults."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_VLLM = REPO_ROOT / "scripts" / "generate_vllm.py"


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributionSweep:
    epsilons: list[float]
    distribution: DistributionSpec | None = None


@dataclass(frozen=True)
class WatermarkSweep:
    distribution_sweeps: list[DistributionSweep] = field(default_factory=list)


WATERMARK_SWEEPS: dict[str, WatermarkSweep] = {

    "NoWatermark": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep( # Dummy sweep with epsilon 0.0 only
                epsilons=[0.0],
                distribution=DistributionSpec("binomial", {"total_count": 1, "probs": 0.5}),
            ),
        ],
    ),

    "KGW": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[1.0, 0.5, 2.0, 3.0, 4.0],
                distribution=DistributionSpec("binomial", {"total_count": 1, "probs": 0.5}),
            ),
        ],
    ),
    "PPLMark": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.0, 0.1, 0.2, 0.3, 0.5],
                distribution=DistributionSpec("binomial", {"total_count": 30, "probs": 0.5}),
            ),
        ],
    ),
    "PPLSingle": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.0, 0.1, 0.2, 0.5, 1.0, 1.5],
                distribution=DistributionSpec("binomial", {"total_count": 30, "probs": 0.5}),
            ),
        ],
    ),
    "Chi2": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.1, 0.5, 1.0, 1.5, 2.0],
                distribution=DistributionSpec("binomial", {"total_count": 30, "probs": 0.5}),
            ),
        ],
    ),
    "AAR": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.0,0.1,0.2,0.3,0.4,0.5],
                distribution=DistributionSpec("gumbel", {"loc": 0, "scale": 1}),
            ),
        ],
    ),
    "SynthID": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[1,2,3,4,5,6],
                distribution=DistributionSpec("binomial", {"total_count": 30, "probs": 0.5}),
            ),
        ],
    ),
}

DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]
DEFAULT_DATASETS = ["sentence-transformers/eli5"]
DEFAULT_N_SAMPLES = 1000
DEFAULT_CONTEXT_SIZES = [4]
DEFAULT_TOP_KS = [50]
DEFAULT_SEEDS = [0]
DEFAULT_RNG_DEVICES = ["cuda"]
DEFAULT_SEEDING_SCHEMES = ["sumhash"]
DEFAULT_QUEUE_CMD = "gpuq"


def available_mapping(watermark_types: Iterable[str]) -> dict[str, dict]:
    """Return a serializable mapping of watermarks to sweeps."""
    mapping = {}
    for wm in watermark_types:
        sweep = WATERMARK_SWEEPS[wm]
        mapping[wm] = {
            "distribution_sweeps": [
                {
                    "epsilons": dist_sweep.epsilons,
                    "distribution": (
                        asdict(dist_sweep.distribution)
                        if dist_sweep.distribution
                        else None
                    ),
                }
                for dist_sweep in sweep.distribution_sweeps
            ]
        }
    return mapping


def build_watermark_configs(
    watermark_type: str,
    sweep: WatermarkSweep,
    rng_devices: list[str],
    seeding_schemes: list[str],
    context_sizes: list[int],
    top_ks: list[int | None],
    seeds: list[int],
) -> Iterator[dict]:
    """Yield concrete watermark configs for each sweep combination."""
    if not sweep.distribution_sweeps:
        raise ValueError(f"No distribution sweeps configured for watermark {watermark_type}")

    for dist_sweep in sweep.distribution_sweeps:
        dist = dist_sweep.distribution
        for epsilon in dist_sweep.epsilons:
            for top_k in top_ks:
                for rng_device in rng_devices:
                    for seeding_scheme in seeding_schemes:
                        for context_size in context_sizes:
                            for seed in seeds:
                                config = {
                                    "epsilon": float(epsilon),
                                    "rng_device": rng_device,
                                    "seeding_scheme": seeding_scheme,
                                    "context_size": int(context_size),
                                    "seed": int(seed),
                                }
                                if top_k is not None:
                                    config["top_k"] = int(top_k)
                                if dist:
                                    config["distribution_name"] = dist.name
                                    config["distribution_parameters"] = dist.parameters
                                yield config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue generate_vllm.py runs via gpuq with watermark sweeps."
    )
    parser.add_argument(
        "--watermark-type",
        "-w",
        action="append",
        dest="watermark_types",
        choices=sorted(WATERMARK_SWEEPS),
        help="Watermark types to sweep. Defaults to all known types.",
    )
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        dest="models",
        help="Model(s) to run. Defaults to meta-llama/Llama-3.1-8B-Instruct.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        dest="datasets",
        help="Dataset(s) to run. Defaults to sentence-transformers/eli5.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per run (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--context-size",
        action="append",
        dest="context_sizes",
        type=int,
        help="Context size(s) to use (default: 4).",
    )
    parser.add_argument(
        "--top-k",
        action="append",
        dest="top_ks",
        type=int,
        help="top_k override(s). Use --omit-top-k to drop this field entirely.",
    )
    parser.add_argument(
        "--omit-top-k",
        action="store_true",
        help="If set, top_k is not passed to generate_vllm.py.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Seed(s) to use (default: 0).",
    )
    parser.add_argument(
        "--rng-device",
        action="append",
        dest="rng_devices",
        help="RNG device(s) to use (default: cuda).",
    )
    parser.add_argument(
        "--seeding-scheme",
        action="append",
        dest="seeding_schemes",
        help="Seeding scheme(s) to use (default: sumhash).",
    )
    parser.add_argument(
        "--queue-cmd",
        default=DEFAULT_QUEUE_CMD,
        help="GPU queue command to use; can be a full command string (e.g., 'gpuq' or 'python ~/gpu_manager.py debug').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for each queue submission to finish. By default, submissions are fire-and-forget.",
    )
    parser.add_argument(
        "--list-mapping",
        action="store_true",
        help="Print the epsilon/distribution mapping and exit.",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable additional metrics computation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Suffix directory to save generated outputs",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--disable-queue",
        action="store_true",
        help="If set, commands are run directly instead of via the queue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    watermark_types = args.watermark_types or sorted(WATERMARK_SWEEPS)
    if args.list_mapping:
        print(json.dumps(available_mapping(watermark_types), indent=2))
        return

    models = args.models or DEFAULT_MODELS
    datasets = args.datasets or DEFAULT_DATASETS
    context_sizes = args.context_sizes or DEFAULT_CONTEXT_SIZES
    seeds = args.seeds or DEFAULT_SEEDS
    rng_devices = args.rng_devices or DEFAULT_RNG_DEVICES
    seeding_schemes = args.seeding_schemes or DEFAULT_SEEDING_SCHEMES
    if args.omit_top_k:
        top_ks: list[int | None] = [None]
    else:
        top_ks = args.top_ks or DEFAULT_TOP_KS

    submitted: list[tuple[str, subprocess.Popen]] = []
    queue_prefix = [os.path.expanduser(part) for part in shlex.split(args.queue_cmd)]

    for model in models:
        for dataset in datasets:
            for wm_type in watermark_types:
                sweep = WATERMARK_SWEEPS[wm_type]
                for config in build_watermark_configs(
                    wm_type,
                    sweep,
                    rng_devices,
                    seeding_schemes,
                    context_sizes,
                    top_ks,
                    seeds,
                ):
                    if args.disable_queue:
                        queue_prefix = []

                    cmd = queue_prefix + [
                        "python",
                        str(GENERATE_VLLM),
                        "--model",
                        model,
                        "--dataset",
                        dataset,
                        "--n_samples",
                        str(args.n_samples),
                        "--watermark-class",
                        wm_type,
                        "--watermark-config",
                        json.dumps(config),
                        "--output_path",
                        args.output_path,
                        "--temperature",
                        str(args.temperature),
                    ]

                    if args.disable_metrics:
                        cmd.append("--disable-metrics")

                    printable_cmd = " ".join(shlex.quote(part) for part in cmd)
                    print(f"Queueing: {printable_cmd}")
                    if args.dry_run:
                        continue
                    if args.disable_queue:
                        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
                    else:
                        proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
                        submitted.append((printable_cmd, proc))

    if args.wait and submitted:
        failures: list[str] = []
        for printable_cmd, proc in submitted:
            if proc.wait() != 0:
                failures.append(printable_cmd)
        if failures:
            raise SystemExit(
                f"Queue submission failed for {len(failures)} job(s):\n"
                + "\n".join(failures)
            )


if __name__ == "__main__":
    main()
