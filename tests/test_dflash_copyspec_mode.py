# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the DFlash copyspec model setting."""

import pytest

from omlx.dflash_runtime import (
    normalize_dflash_copyspec_mode,
    with_optional_copyspec_mode,
)
from omlx.model_profiles import MODEL_SPECIFIC_PROFILE_FIELDS
from omlx.model_settings import ModelSettings, ModelSettingsManager


def test_copyspec_default_stays_unset_and_uses_runtime_default():
    settings = ModelSettings()

    assert settings.dflash_copyspec_mode is None
    assert "dflash_copyspec_mode" not in settings.to_dict()

    def current_runtime_config(*, copyspec_mode: str = "conservative") -> str:
        return copyspec_mode

    kwargs = with_optional_copyspec_mode(current_runtime_config, {}, None)
    assert kwargs == {}
    assert current_runtime_config(**kwargs) == "conservative"


def test_copyspec_mode_round_trips_and_is_profile_scoped():
    settings = ModelSettings.from_dict({"dflash_copyspec_mode": "auto"})

    assert settings.dflash_copyspec_mode == "auto"
    assert settings.to_dict()["dflash_copyspec_mode"] == "auto"
    assert "dflash_copyspec_mode" in MODEL_SPECIFIC_PROFILE_FIELDS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", "auto"),
        ("off", "off"),
        ("unsupported", None),
        ("AUTO", None),
        (1, None),
    ],
)
def test_direct_json_normalizes_copyspec_mode_at_model_settings_boundary(
    value, expected
):
    settings = ModelSettings.from_dict({"dflash_copyspec_mode": value})

    assert settings.dflash_copyspec_mode == expected
    assert settings.to_dict().get("dflash_copyspec_mode") == expected


@pytest.mark.parametrize(
    ("value", "expected"), [("auto", "auto"), ("unsupported", None)]
)
def test_profile_application_normalizes_copyspec_mode(tmp_path, value, expected):
    manager = ModelSettingsManager(tmp_path)
    manager.save_profile(
        model_id="model-a",
        name="copy",
        display_name="Copy",
        description=None,
        settings={"dflash_copyspec_mode": value},
    )

    applied = manager.apply_profile("model-a", "copy")

    assert applied is not None
    assert applied.dflash_copyspec_mode == expected


@pytest.mark.parametrize("value", [None, "", "invalid", "AUTO", 1])
def test_invalid_copyspec_modes_revert_to_dflash_default(value):
    assert normalize_dflash_copyspec_mode(value) is None


@pytest.mark.parametrize("value", ["conservative", "auto", "off"])
def test_valid_copyspec_modes_are_preserved(value):
    assert normalize_dflash_copyspec_mode(value) == value


def test_runtime_config_receives_copyspec_mode_when_supported():
    seen = {}

    def current_runtime_config(*, copyspec_mode=None, **kwargs):
        seen.update(kwargs)
        seen["copyspec_mode"] = copyspec_mode

    kwargs = with_optional_copyspec_mode(
        current_runtime_config,
        {"verify_mode": "adaptive"},
        "auto",
    )
    current_runtime_config(**kwargs)

    assert seen == {"verify_mode": "adaptive", "copyspec_mode": "auto"}


def test_runtime_config_omits_copyspec_mode_when_dflash_lacks_support():
    def legacy_runtime_config(*, verify_mode=None):
        return verify_mode

    original_kwargs = {"verify_mode": "adaptive"}
    kwargs = with_optional_copyspec_mode(
        legacy_runtime_config,
        original_kwargs,
        "auto",
    )

    assert kwargs is original_kwargs
    assert "copyspec_mode" not in kwargs
    assert legacy_runtime_config(**kwargs) == "adaptive"
