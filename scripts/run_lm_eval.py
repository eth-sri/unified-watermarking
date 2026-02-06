#!/usr/bin/env python
"""Sweep watermark configs and run lm_eval benchmarks against a vLLM server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

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
            DistributionSweep(
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
                epsilons=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                distribution=DistributionSpec("gumbel", {"loc": 0, "scale": 1}),
            ),
        ],
    ),
    "SynthID": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[1, 2, 3, 4, 5, 6],
                distribution=DistributionSpec("binomial", {"total_count": 30, "probs": 0.5}),
            ),
        ],
    ),
}

DEFAULT_LM_EVAL_MODEL = "local-completions"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.2:8000/v1/completions"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_NUM_CONCURRENT = 8
DEFAULT_MAX_GEN_TOKS = 200
DEFAULT_TASKS = "gsm8k,humaneval,mbpp"
DEFAULT_OUTPUT_DIR = "output/lm_eval"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_VOCAB_SIZE = 128256
DEFAULT_CONTEXT_SIZES = [4]
DEFAULT_TOP_KS = [50]
DEFAULT_SEEDS = [0]
DEFAULT_RNG_DEVICES = ["cuda"]
DEFAULT_SEEDING_SCHEMES = ["sumhash"]


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
    sweep: WatermarkSweep,
    rng_devices: list[str],
    seeding_schemes: list[str],
    context_sizes: list[int],
    top_ks: list[int | None],
    seeds: list[int],
) -> Iterator[dict]:
    """Yield concrete watermark configs for each sweep combination."""
    if not sweep.distribution_sweeps:
        raise ValueError("No distribution sweeps configured.")

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


def normalize_distribution_parameters(value: object | None) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def build_vllm_xargs(
    watermark_type: str,
    config: dict,
    vocab_size: int,
) -> dict:
    vllm_xargs = dict(config)
    vllm_xargs["watermark_class"] = watermark_type
    vllm_xargs["vocab_size"] = vocab_size
    if "distribution_parameters" in vllm_xargs:
        dist_params = normalize_distribution_parameters(
            vllm_xargs.get("distribution_parameters")
        )
        if dist_params is None:
            vllm_xargs.pop("distribution_parameters", None)
        else:
            vllm_xargs["distribution_parameters"] = dist_params
    return vllm_xargs


def build_gen_kwargs(temperature: float, vllm_xargs: dict) -> dict:
    gen_kwargs = {
        "temperature": temperature,
        "do_sample": True,
        "vllm_xargs": vllm_xargs,
    }
    if "top_k" in vllm_xargs:
        gen_kwargs["top_k"] = vllm_xargs["top_k"]
    return gen_kwargs


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def slugify(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def format_config_slug(config: dict) -> str:
    parts = [
        f"eps{slugify(config.get('epsilon'))}",
        f"rng{slugify(config.get('rng_device'))}",
        f"scheme{slugify(config.get('seeding_scheme'))}",
        f"ctx{slugify(config.get('context_size'))}",
        f"seed{slugify(config.get('seed'))}",
    ]
    if "top_k" in config:
        parts.append(f"topk{slugify(config.get('top_k'))}")
    if "distribution_name" in config:
        parts.append(f"dist{slugify(config.get('distribution_name'))}")
    parts.append(config_hash(config))
    return "_".join(parts)


def build_model_args(args: argparse.Namespace) -> str:
    if args.model_args:
        return args.model_args
    return ",".join(
        [
            f"model={args.model_name}",
            f"base_url={args.base_url}",
            f"api_key={args.api_key}",
            f"num_concurrent={args.num_concurrent}",
            f"max_gen_toks={args.max_gen_toks}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep watermark configs and run lm_eval benchmarks. Make sure the vLLM server is running.",
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
        "--tasks",
        "-t",
        default=DEFAULT_TASKS,
        help="Comma-separated list of lm_eval tasks (default: gsm8k).",
    )
    parser.add_argument(
        "--lm-eval-model",
        default=DEFAULT_LM_EVAL_MODEL,
        help="lm_eval --model value (default: local-completions).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Served model name used in lm_eval --model_args.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="vLLM completions endpoint base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the vLLM server.",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=DEFAULT_NUM_CONCURRENT,
        help="Concurrent requests for local-completions.",
    )
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=DEFAULT_MAX_GEN_TOKS,
        help="Maximum generation tokens for local-completions.",
    )
    parser.add_argument(
        "--model-args",
        default=None,
        help="Raw lm_eval --model_args string. Overrides the individual model args.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for lm_eval outputs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id for grouping outputs. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Vocabulary size to pass in vllm_xargs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for lm_eval generation.",
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
        help="If set, top_k is not passed in gen_kwargs or vllm_xargs.",
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
        "--batch-size",
        default=None,
        help="lm_eval --batch_size value (e.g. auto or 16).",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="lm_eval --limit value.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="lm_eval --num_fewshot value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="lm_eval --device value (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Pass --log_samples to lm_eval and save per-sample outputs.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Pass --predict_only to lm_eval (implies --log_samples).",
    )
    parser.add_argument(
        "--queue-cmd",
        default="",
        help="Optional queue command prefix (e.g. 'gpuq' or 'python ~/gpu_manager.py debug').",
    )
    parser.add_argument(
        "--confirm-run-unsafe-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --confirm_run_unsafe_code to lm_eval (default: true).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--list-mapping",
        action="store_true",
        help="Print the epsilon/distribution mapping and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    watermark_types = args.watermark_types or sorted(WATERMARK_SWEEPS)
    if args.list_mapping:
        print(json.dumps(available_mapping(watermark_types), indent=2))
        return

    context_sizes = args.context_sizes or DEFAULT_CONTEXT_SIZES
    seeds = args.seeds or DEFAULT_SEEDS
    rng_devices = args.rng_devices or DEFAULT_RNG_DEVICES
    seeding_schemes = args.seeding_schemes or DEFAULT_SEEDING_SCHEMES
    if args.omit_top_k:
        top_ks: list[int | None] = [None]
    else:
        top_ks = args.top_ks or DEFAULT_TOP_KS

    model_args = build_model_args(args)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / run_id

    queue_prefix = [os.path.expanduser(part) for part in shlex.split(args.queue_cmd)]

    for wm_type in watermark_types:
        sweep = WATERMARK_SWEEPS[wm_type]
        for config in build_watermark_configs(
            sweep,
            rng_devices,
            seeding_schemes,
            context_sizes,
            top_ks,
            seeds,
        ):
            vllm_xargs = build_vllm_xargs(wm_type, config, args.vocab_size)
            gen_kwargs = build_gen_kwargs(args.temperature, vllm_xargs)
            run_slug = format_config_slug(config)
            run_dir = output_root / wm_type / run_slug
            run_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "gen_kwargs.json").write_text(
                json.dumps(gen_kwargs, indent=2, sort_keys=True)
            )
            (run_dir / "watermark_config.json").write_text(
                json.dumps(vllm_xargs, indent=2, sort_keys=True)
            )

            output_path = str(run_dir / "results.json")

            cmd = queue_prefix + [
                "lm_eval",
                "--model",
                args.lm_eval_model,
                "--model_args",
                model_args,
                "--tasks",
                args.tasks,
                "--gen_kwargs",
                json.dumps(gen_kwargs, separators=(",", ":")),
                "--output_path",
                output_path,
            ]

            if args.batch_size:
                cmd += ["--batch_size", str(args.batch_size)]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if args.num_fewshot is not None:
                cmd += ["--num_fewshot", str(args.num_fewshot)]
            if args.device:
                cmd += ["--device", args.device]
            if args.log_samples:
                cmd.append("--log_samples")
            if args.predict_only:
                cmd.append("--predict_only")
            if args.confirm_run_unsafe_code:
                cmd.append("--confirm_run_unsafe_code")

            (run_dir / "lm_eval_cmd.txt").write_text(
                " ".join(shlex.quote(part) for part in cmd) + "\n"
            )

            printable_cmd = " ".join(shlex.quote(part) for part in cmd)
            print(f"Running: {printable_cmd}")
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
