# A Unified Framework for LLM Watermarks

This repository provides a unified framework for LLM watermarking. It supports
existing schemes and makes it straightforward to implement new ones.

## Installation

The package requires Python 3.9+.

We recommend using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Overview

Watermarking is integrated through a vLLM logits processor for fast inference.

Key locations:
- Implemented watermark classes: `src/lm_wm_tools/watermarks`
- Main package interface: `src/lm_wm_tools/__init__.py`
- Logits processor and factory helpers: `src/lm_wm_tools/watermarks/__init__.py`

## Generating and Detecting Watermarked Text

Watermarks are applied through a single vLLM logits processor.
- Offline (`vllm.LLM`): pass watermark config through `SamplingParams.extra_args`
- Online (`vllm serve`): pass watermark config through `vllm_xargs`

### Online

Start a server with the watermark logits processor:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --logits-processors lm_wm_tools.watermarks:WatermarkLogitsProcessor \
  --generation-config vllm
```

When sending a request, include the watermark config in `vllm_xargs`.

`vllm_xargs` does not support nested dictionaries. For nested values (for
example, `distribution_parameters`), pass a JSON string and let the logits
processor decode it.

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

watermark_config = {
    "watermark_class": "KGW",
    "epsilon": 1.0,
    "vocab_size": 128256,
    "rng_device": "cuda",
    "seeding_scheme": "sumhash",
    "context_size": 4,
    "seed": 0,
    "top_k": 50,
    "distribution_name": "binomial",
    "distribution_parameters": json.dumps({"total_count": 1, "probs": 0.5}),
}

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Write a small story about a brave knight."},
    ],
    extra_body={
        "top_k": 50,
        "vllm_xargs": watermark_config,
    },
)

print(resp.choices[0].message.content)
```

If the watermark config is invalid, request processing can fail.

### Offline

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from lm_wm_tools.watermarks import WatermarkLogitsProcessor

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

watermark_config = {
    "watermark_class": "KGW",  # short name or full import path
    "epsilon": 1.0,
    "vocab_size": len(tokenizer.get_vocab()),
    "rng_device": "cuda",
    "seeding_scheme": "sumhash",
    "context_size": 4,
    "seed": 0,
    "top_k": 50,
    "distribution_name": "binomial",
    "distribution_parameters": {"total_count": 1, "probs": 0.5},
}

llm = LLM(model=model_name, logits_processors=[WatermarkLogitsProcessor])
params = SamplingParams(
    temperature=0.7,
    max_tokens=200,
    top_k=watermark_config.get("top_k", -1),
    extra_args=watermark_config,
)

outputs = llm.generate(["Write a short story about a rabbit."], sampling_params=params)
```

### Detection

You can instantiate a watermark directly and run detection on token IDs.

Use the same sampling parameters that were used for generation if your
watermark relies on them during detection.

```python
from vllm import SamplingParams
from lm_wm_tools.watermarks import get_watermark

# Use your sampling parameters
params = SamplingParams(
    temperature=0.7,
    max_tokens=200,
    top_k=watermark_config.get("top_k", -1),
    extra_args=watermark_config,
)

watermark = get_watermark(watermark_config, params)
token_ids = outputs[0].outputs[0].token_ids
scores = watermark.detect(token_ids)
```

Detection typically returns a dictionary with `statistic` and `pvalue` (and
sometimes extra fields). See each watermark class for details.

### CLI Generation

For scripted runs, `scripts/generate_vllm.py` accepts a JSON watermark config:

```bash
python scripts/generate_vllm.py \
  --watermark-class KGW \
  --watermark-config '{"epsilon":1.0,"rng_device":"cuda","seeding_scheme":"sumhash","context_size":4,"seed":0}'
```

`--watermark-config` can also be a path to a JSON file.

## Advanced: Logits Metrics

If you pass `request_id` and `metrics_dir` in `extra_args`, the logits
processor wraps your watermark with `LogitsMetricWrapper` and writes per-step
metrics to `metrics_dir/<request_id>.jsonl`.

```python
import uuid
from lm_wm_tools.watermarks.metrics import summarize_metrics

req_id = str(uuid.uuid4())
metrics_dir = "output/metrics"
watermark_config.update({"request_id": req_id, "metrics_dir": metrics_dir})

metrics = summarize_metrics(req_id, metrics_dir)
```

Metrics include:
- `kl_divergence`, `chi2_divergence`
- `ppl_hard`, `ppl_soft`
- `kl_divergence_soft`

To make `ppl_soft` and `kl_divergence_soft` meaningful, implement
`get_expected_probs` in your watermark; otherwise, it defaults to the original
probabilities.

## Advanced: Quality Metrics

`lm_wm_tools.metrics.quality_metrics` provides helper functions for output
quality and diversity:

```python
from lm_wm_tools.metrics.quality_metrics import compute_perplexity, compute_self_bleu

perplexities = compute_perplexity(model, input_ids, attention_mask)
self_bleu = compute_self_bleu(generated_sequences)
```
