import os
from vllm import LLM, SamplingParams
from lm_wm_tools.watermarks import WatermarkLogitsProcessor, get_watermark
from lm_wm_tools.watermarks.metrics import summarize_metrics
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import argparse
import json
import sys
import uuid
import tempfile

from loguru import logger

OUTPUT_PATH = "output/llm_completions"

logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

class DatasetNotFoundError(Exception):
    """Raised when the dataset name is not recognized."""


class DatasetLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _process_chat_dataset(self, dataset: Dataset, n_samples: int, n_existing_samples: int, question_key: str = "question") -> list:
        dataset = dataset.skip(n_existing_samples)
        prompts_ds = []
        for row in dataset:
            text = row[question_key]
            prompts_ds.append([{"role": "user", "content": text}])
            if len(prompts_ds) >= n_samples:
                break
        return prompts_ds

    def _process_completion_dataset(self, dataset: Dataset, n_samples: int, n_existing_samples: int, text_key: str = "text") -> list:
        dataset = dataset.skip(n_existing_samples)
        prompts_ds = []
        for row in dataset:
            text = row[text_key]
            tokenized_text = self.tokenizer(text)
            if len(tokenized_text["input_ids"]) < 200:
                continue
            text = self.tokenizer.decode(tokenized_text["input_ids"][:200])
            prompts_ds.append(text)
            if len(prompts_ds) >= n_samples:
                break
        return prompts_ds

    def load(self, dataset_name: str, n_samples: int, n_existing_samples: int):
        if "c4" in dataset_name:
            is_chat = False
            ds = load_dataset(
                dataset_name, name="realnewslike", split="validation", streaming=False
            )
            prompts_ds = self._process_completion_dataset(ds, n_samples, n_existing_samples)
            return prompts_ds, is_chat
        if "eli5" in dataset_name:
            is_chat = True
            ds = load_dataset(
                dataset_name, split="train", streaming=False
            )
            prompts_ds = self._process_chat_dataset(ds, n_samples, n_existing_samples)
            return prompts_ds, is_chat

        if dataset_name == "diversity_eval":
            is_chat = True

            logger.warning("Diversity evaluation overrides the number of samples.")

            
            ds = load_dataset(
                "sentence-transformers/eli5", split="train", streaming=False
            )
            prompts_ds = self._process_chat_dataset(ds, n_samples=100, n_existing_samples=0)

            # Duplicate each prompt 100 times to get 10,000 samples
            expanded_prompts_ds = []
            for prompt in prompts_ds:
                for _ in range(100):
                    expanded_prompts_ds.append(prompt)
            
            prompts_ds = expanded_prompts_ds
            return prompts_ds, is_chat


        raise DatasetNotFoundError(f"Unknown dataset: {dataset_name}")


def parse_watermark_config(value: str) -> dict:
    """Allow passing either an inline JSON string or a path to a JSON file."""
    if os.path.exists(value):
        try:
            with open(value, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"Failed to decode JSON watermark config from file {value}: {exc}"
            ) from exc
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Failed to decode JSON watermark config from string: {exc}"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate watermarking detection")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--watermark_class",
        "--watermark-class",
        type=str,
        required=True,
        help="Watermark class to use",
    )
    parser.add_argument(
        "--watermark-config",
        "--watermark_config",
        dest="watermark_config",
        type=parse_watermark_config,
        required=True,
        help="Configuration parameters for the watermarking scheme",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/c4",
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to use for evaluation",
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
    return parser.parse_args()


def main(args):
    global OUTPUT_PATH

    if args.output_path:
        OUTPUT_PATH = f"{OUTPUT_PATH}/{args.output_path}"

    model_to_load = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)

    watermark_parameters = dict(args.watermark_config)
    watermark_parameters["watermark_class"] = args.watermark_class
    watermark_parameters.setdefault("vocab_size", len(tokenizer.get_vocab()))
    for key, value in watermark_parameters.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    logits_processor = WatermarkLogitsProcessor

    sampling_parameters = SamplingParams(
        temperature=args.temperature,
        max_tokens=200,
        min_tokens=200,
        top_k=watermark_parameters.get("top_k", -1),
        extra_args=watermark_parameters,
    )

    watermark = get_watermark(watermark_parameters, sampling_parameters)


    # Override the output path
    if args.dataset == "diversity_eval":
        OUTPUT_PATH = f"{OUTPUT_PATH}-diversity"


    # Check how many samples have already been generated
    # We count the number of rows with exactly the same watermark parameters + dataset
    path = f"{OUTPUT_PATH}/{watermark.get_name()}/completions.jsonl"
    n_existing_samples = 0
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                example = json.loads(line)
                match = True
                for key, value in watermark_parameters.items():
                    if example.get(key) != value:
                        match = False
                        break
                if example.get("dataset") != args.dataset:
                    match = False
                if match:
                    n_existing_samples += 1
    logger.info(f"Found {n_existing_samples} existing samples with the same config.")
    if n_existing_samples >= args.n_samples:
        logger.info("No new samples to generate. Exiting.")
        return
    args.n_samples -= n_existing_samples
    logger.info(f"Generating {args.n_samples} new samples.")

    lm = LLM(
        model_to_load,
        logits_processors=[logits_processor],
        max_num_seqs=64,
    )
    dataset_loader = DatasetLoader(tokenizer)
    prompts_ds, is_chat = dataset_loader.load(
        args.dataset, args.n_samples, n_existing_samples
    )

    batch_sampling_params = []
    batch_request_ids = []
   
    with tempfile.TemporaryDirectory() as temp_dir:

        for _ in prompts_ds:
            req_id = str(uuid.uuid4())
            batch_request_ids.append(req_id)
            
            # For additional metrics collection, we pass a unique request_id and metrics_dir
            req_wm_params = watermark_parameters.copy()
            if not args.disable_metrics:
                req_wm_params["request_id"] = req_id
                req_wm_params["metrics_dir"] = temp_dir

            top_k = watermark_parameters.get("top_k", -1)

            sp = SamplingParams(
                temperature=args.temperature,
                max_tokens=200,
                min_tokens=10,
                top_k=top_k,
                extra_args=req_wm_params,
            )
            batch_sampling_params.append(sp)

        if is_chat:
            outputs = lm.chat(messages=prompts_ds, sampling_params=batch_sampling_params)
        else:
            outputs = lm.generate(prompts_ds, sampling_params=batch_sampling_params)


        # Collect prompts, generated text, and detection scores
        examples = []
        for prompt, output, req_id in zip(prompts_ds, outputs, batch_request_ids):            
            metrics = summarize_metrics(req_id, temp_dir)

            candidate = output.outputs[0]
            detection_scores = watermark.detect(tokens=candidate.token_ids)
            example = {
                "prompt": prompt,
                "dataset": args.dataset,
                "output_text": candidate.text,
                "output_length": len(candidate.token_ids),
                **detection_scores,
                **metrics,
            }
            examples.append(example)
        # If the file exists, append to it
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                pass
        with open(path, "a") as f:
            for example in examples:
                example.update(watermark_parameters)
                f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
