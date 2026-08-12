"""Regression tests for admin model-settings UI gates."""

from pathlib import Path


def _model_settings_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()


def _dashboard_js() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "omlx/admin/static/js/dashboard.js").read_text()


def _section(html: str, start_marker: str, end_marker: str) -> str:
    return html.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_lightning_mtp_and_turboquant_are_not_ui_mutexed():
    html = _model_settings_template()

    turboquant = _section(
        html,
        "<!-- TurboQuant KV Cache -->",
        "<!-- IndexCache (DSA models only) -->",
    )
    lightning_mtp = _section(
        html,
        "<!-- Lightning MTP (built-in MTP head speculative decoding) -->",
        "<!-- Experimental Features -->",
    )

    assert "modelSettings.mtp_enabled" not in turboquant
    assert "modelSettings.turboquant_kv_enabled" not in lightning_mtp


def test_vlm_mtp_still_conflicts_with_turboquant():
    html = _model_settings_template()
    vlm_mtp = _section(
        html,
        "<!-- VLM MTP",
        "<!-- Performance",
    )

    assert "modelSettings.turboquant_kv_enabled" in vlm_mtp


def test_reasoning_effort_offers_max_after_high():
    html = _model_settings_template()

    high_option = '<option value="high">'
    max_option = '<option value="max">max</option>'

    assert high_option in html
    assert max_option in html
    assert html.index(high_option) < html.index(max_option)


def test_dflash_copyspec_mode_has_all_runtime_options():
    html = _model_settings_template()

    assert 'x-model="modelSettings.dflash_copyspec_mode"' in html
    assert '<option value="conservative">conservative (default)</option>' in html
    assert '<option value="auto">auto</option>' in html
    assert '<option value="off">off</option>' in html


def test_dflash_copyspec_mode_uses_and_submits_the_conservative_default():
    dashboard = _dashboard_js()

    assert "dflash_copyspec_mode: s.dflash_copyspec_mode || 'conservative'" in dashboard
    assert "? (this.modelSettings.dflash_copyspec_mode || 'conservative')" in dashboard
