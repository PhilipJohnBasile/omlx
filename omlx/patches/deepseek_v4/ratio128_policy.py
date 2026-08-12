# SPDX-License-Identifier: Apache-2.0
"""Pure-Python routing policy for DeepSeek-V4 ratio-128 attention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _contains_sub4_bits(value: Any) -> bool:
    """Return whether a quantization mapping contains a numeric sub-4 bit mode."""
    if isinstance(value, Mapping):
        bits = value.get("bits")
        if (
            isinstance(bits, (int, float))
            and not isinstance(bits, bool)
            and float(bits) < 4
        ):
            return True
        return any(_contains_sub4_bits(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_sub4_bits(child) for child in value)
    return False


def has_sub4_quantization(config: Mapping[str, Any]) -> bool:
    """Return whether any supported DeepSeek-V4 quantization section is sub-4-bit.

    oMLX quantization configs may carry a high-bit default with lower-bit
    per-module overrides, so checking only the root ``bits`` field is unsafe.
    """
    quantizations = [config.get("quantization"), config.get("quantization_config")]
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        quantizations.append(text_config.get("quantization_config"))
    return any(_contains_sub4_bits(quantization) for quantization in quantizations)


def native_ratio128_attention_enabled(config: Mapping[str, Any]) -> bool:
    """Keep native ratio-128 attention off for sub-4-bit DeepSeek-V4 models."""
    if not str(config.get("model_type", "")).startswith("deepseek_v4"):
        return True
    return not has_sub4_quantization(config)


def should_attempt_native_ratio128_attention(
    *,
    enabled: bool,
    compress_ratio: int,
    standard_mask: bool,
    has_pooled_rows: bool,
    query_length: int,
    scalar_cache_offset: bool,
    dspark_single_token_decode: bool,
) -> bool:
    """Return whether the native ratio-128 kernel can preserve dense semantics."""
    return (
        enabled
        and compress_ratio == 128
        and standard_mask
        and has_pooled_rows
        and query_length > 4
        and scalar_cache_offset
        and not dspark_single_token_decode
    )
