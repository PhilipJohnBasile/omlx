# SPDX-License-Identifier: Apache-2.0
"""Configuration-only admission rules for speculative decoding.

The helpers here deliberately inspect JSON-compatible configuration only.  They
must stay independent of MLX so model discovery and load admission can fail
closed without allocating a device array or importing an inference backend.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_QWEN36_DENSE_27B_SIGNATURE = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "full_attention_interval": 4,
}


def qwen36_affine_q8_g64_speculative_block_reason(
    config: Mapping[str, Any],
) -> str | None:
    """Return the active #2508 block reason, or ``None`` when it does not apply.

    Qwen3.6's public config uses the shared ``qwen3_5`` model type, so a
    model-type check alone would overblock Qwen3.5 and Qwen3.6 MoE targets.
    The fixed dense-27B text geometry below is the narrow, published target
    from the reproducible report.  Q8 with no explicit mode is included:
    MLX treats that legacy representation as affine quantization.

    This is an admission guard, not a numerical repair.  It applies only to
    MTP target verification; callers must not generalize it to unrelated
    quantized decode paths without separate evidence.
    """
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        return None
    if config.get("model_type") != "qwen3_5":
        return None
    if text_config.get("model_type") != "qwen3_5_text":
        return None
    if any(
        text_config.get(key) != value
        for key, value in _QWEN36_DENSE_27B_SIGNATURE.items()
    ):
        return None

    quantization = config.get("quantization")
    if not isinstance(quantization, Mapping):
        quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        return None
    if quantization.get("bits") != 8 or quantization.get("group_size") != 64:
        return None

    mode = quantization.get("mode")
    if mode is not None and str(mode).lower() != "affine":
        return None

    return (
        "#2508 safety gate: Qwen3.6 dense 27B affine Q8/group-64 targets "
        "are not admitted to multi-token MTP verification because greedy "
        "output divergence is reproduced; use ordinary decode."
    )


def speculative_verification_allowed(config: Mapping[str, Any]) -> bool:
    """Whether this target may enter an MTP multi-token verify loop."""
    return qwen36_affine_q8_g64_speculative_block_reason(config) is None


def speculative_verification_block_reason_from_path(
    model_path: str | Path,
) -> str | None:
    """Read a local ``config.json`` for load admission without touching weights.

    An unreadable or malformed config has no established #2508 classification,
    so this helper returns ``None``.  Normal loader validation continues to
    own malformed-checkpoint errors.
    """
    try:
        payload = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return qwen36_affine_q8_g64_speculative_block_reason(payload)
