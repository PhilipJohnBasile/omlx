# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free compatibility helpers for DFlash runtime settings."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

DFLASH_COPYSPEC_MODES = frozenset(("conservative", "auto", "off"))


def normalize_dflash_copyspec_mode(value: object) -> str | None:
    """Return a supported copyspec policy, or ``None`` for the DFlash default."""
    return value if isinstance(value, str) and value in DFLASH_COPYSPEC_MODES else None


def with_optional_copyspec_mode(
    runtime_config_from_defaults: Callable[..., object],
    kwargs: dict[str, Any],
    copyspec_mode: str | None,
) -> dict[str, Any]:
    """Add ``copyspec_mode`` only when this DFlash runtime supports it.

    Leaving an unset value out preserves the runtime's own default.  Signature
    inspection also keeps oMLX usable with an older dflash-mlx installation
    whose ``runtime_config_from_defaults`` has no copyspec argument.
    """
    if copyspec_mode is None:
        return kwargs
    try:
        parameters = inspect.signature(runtime_config_from_defaults).parameters
    except (TypeError, ValueError):
        return kwargs
    if "copyspec_mode" not in parameters and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return kwargs
    return {**kwargs, "copyspec_mode": copyspec_mode}
