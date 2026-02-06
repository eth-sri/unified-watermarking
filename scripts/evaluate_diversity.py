import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from lm_wm_tools.metrics.quality_metrics import compute_self_bleu
from loguru import logger
import sys

logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the diversity of generated completions."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Tokenizer to use for encoding completions.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a directory containing completion jsonl files."
    )
    return parser.parse_args()


def main(args):
    tokenizer_name = args.tokenizer

    input_path = Path(args.input_path)
    input_paths = list(input_path.glob("**/completions.jsonl"))
    n_inputs = len(input_paths)

    logger.info(f"Found {n_inputs} completion files to process.")

    if n_inputs == 0:
        raise ValueError(f"No completion files found in {input_path}.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    for completion_file in input_paths:
        df = pd.read_json(completion_file, lines=True)

        dataset_mask = df["dataset"] == "diversity_eval"
        if not dataset_mask.any():
            logger.warning(
                f"No diversity_eval dataset found in {completion_file}, "
                "skipping."
            )
            continue

        if "output_text" not in df.columns:
            raise KeyError(
                f"Expected column 'output_text' in {completion_file}, "
                "but it was not found."
            )

        if "prompt" not in df.columns:
            raise KeyError(
                f"Expected column 'prompt' in {completion_file}, "
                "but it was not found."
            )

        weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}

        # Add columns for self-bleu scores to the FULL df (others remain NaN)
        for key in weights.keys():
            col = f"self_bleu_{key}"
            if col not in df.columns:
                df[col] = np.nan

        # Compute self-bleu ONLY for diversity_eval rows, grouped by prompt
        df_div = df.loc[dataset_mask]  # keep original indices for assignment back to df
        df_div["prompt"] = df_div["prompt"].astype(str) # ensure prompt is str for grouping
        df_groupby_prompt = df_div.groupby("prompt")

        for _, group in df_groupby_prompt:
            completions = group["output_text"].tolist()
            encodings = tokenizer(
                completions,
                padding=False,
            )["input_ids"]

            bleu_score = compute_self_bleu(encodings)

            for key, value in bleu_score.items():
                df.loc[group.index, f"self_bleu_{key}"] = value

        save_path = completion_file
        df.to_json(save_path, orient="records", lines=True)
        logger.info(f"Saved results with diversity (self-bleu) to {save_path}.")


if __name__ == "__main__":
    main(parse_args())
