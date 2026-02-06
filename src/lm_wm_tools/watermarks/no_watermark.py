"""Dummy Watermark For Easier Integration."""

from lm_wm_tools import WatermarkProtocol

class DummyWatermark(WatermarkProtocol):
  """SynthID watermarking implementation for vLLM logits processors."""

  def __init__(
      self,
      **kwargs,
  ):
    pass
    
  def get_name(self) -> str:
    return "Unwatermarked"

  def __call__(
      self,
      output_ids: list[int],
      logits,
  ):
    return logits


  def detect(
      self,
      **kwargs,
  ) -> dict:
    return {}
