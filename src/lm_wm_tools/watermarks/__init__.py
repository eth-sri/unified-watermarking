import copy
import importlib
from typing import Optional
import json

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.metrics import LogitsMetricWrapper

_WATERMARK_MAPPING = {
    "KGW": "lm_wm_tools.watermarks.kgw.red_green.RedGreenWatermark",
    "Chi2": "lm_wm_tools.watermarks.chi2.chitwo.ChiTwoWatermark",
    "AAR": "lm_wm_tools.watermarks.aar_extended.aar.AARWatermark",
    "PPLMark": "lm_wm_tools.watermarks.aar_extended.ppl_wm.PPLWatermark",
    "PPLSingle": "lm_wm_tools.watermarks.ppl.ppl_single_fast.PPLSingleFastWatermark",
    "SynthID": "lm_wm_tools.watermarks.synthid.synthid.SynthIDWatermark",
    "NoWatermark": "lm_wm_tools.watermarks.no_watermark.DummyWatermark",
}
 

# Wrapping the request-level logits processor:
class WatermarkLogitsProcessor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        
        # Extract config
        watermark_config = copy.deepcopy(params.extra_args)

        # Extract our special side-channel keys
        request_id = watermark_config.pop("request_id", None)
        metrics_dir = watermark_config.pop("metrics_dir", None)

        # Parse distribution_parameters
        distribution_parameters = watermark_config.pop("distribution_parameters", {})
        if isinstance(distribution_parameters, str):
            distribution_parameters = json.loads(distribution_parameters)
        watermark_config["distribution_parameters"] = distribution_parameters
        
        watermark_class = watermark_config.pop("watermark_class", None)
        assert watermark_class is not None, "watermark_class missing"

        watermark_processor = get_watermark_class(watermark_class)(sampling_parameters=params, **watermark_config)
        if request_id is not None and metrics_dir is not None:
            return LogitsMetricWrapper(watermark_processor, request_id, metrics_dir, params.temperature)
        else:
            return watermark_processor

def get_watermark_class(spec: str) -> type[WatermarkProtocol]:

    full_path = _WATERMARK_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = f"Unknown watermark type: {spec} (resolved to {full_path}, available: {_WATERMARK_MAPPING})"
        raise ValueError(msg) from e


def get_watermark(watermark_config: dict, sampling_params: SamplingParams) -> WatermarkProtocol:
    config = copy.deepcopy(watermark_config)
    watermark_class = config.pop("watermark_class", None)

    # Parse distribution_parameters
    distribution_parameters = config.pop("distribution_parameters", {})
    if isinstance(distribution_parameters, str):
        distribution_parameters = json.loads(distribution_parameters)
    config["distribution_parameters"] = distribution_parameters


    assert watermark_class is not None, "watermark_class must be specified in watermark_config"
    return get_watermark_class(watermark_class)(sampling_parameters=sampling_params, **config)
