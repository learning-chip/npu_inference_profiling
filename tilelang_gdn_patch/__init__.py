"""TileLang GDN (chunk) reference implementation + benchmarks vs vLLM Triton."""

from .api import chunk_gated_delta_rule_tilelang

__all__ = ["chunk_gated_delta_rule_tilelang"]
