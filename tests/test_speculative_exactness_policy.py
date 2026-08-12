# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission tests for known speculative exactness hazards."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from omlx.speculative.exactness_policy import (
    qwen36_affine_q8_g64_speculative_block_reason,
    speculative_verification_allowed,
    speculative_verification_block_reason_from_path,
)


def _qwen36_dense_27b_config(**quantization):
    return {
        "model_type": "qwen3_5",
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"} | quantization,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "full_attention_interval": 4,
        },
    }


def test_qwen36_affine_q8_g64_is_blocked_by_production_predicate():
    config = _qwen36_dense_27b_config()

    reason = qwen36_affine_q8_g64_speculative_block_reason(config)

    assert reason is not None
    assert "#2508" in reason
    assert not speculative_verification_allowed(config)


@pytest.mark.parametrize(
    "change",
    [
        {"bits": 6},
        {"group_size": 32},
        {"mode": "mxfp8"},
    ],
)
def test_other_quantization_modes_are_not_overblocked(change):
    config = _qwen36_dense_27b_config(**change)

    assert qwen36_affine_q8_g64_speculative_block_reason(config) is None
    assert speculative_verification_allowed(config)


def test_legacy_q8_without_mode_is_blocked_as_affine_default():
    config = _qwen36_dense_27b_config()
    del config["quantization"]["mode"]

    assert qwen36_affine_q8_g64_speculative_block_reason(config) is not None


def test_qwen3_5_moe_q8_is_not_conflated_with_dense_qwen36():
    config = _qwen36_dense_27b_config()
    config["model_type"] = "qwen3_5_moe"
    config["text_config"]["model_type"] = "qwen3_5_moe_text"

    assert qwen36_affine_q8_g64_speculative_block_reason(config) is None


def test_other_dense_qwen_geometry_is_not_overblocked():
    config = _qwen36_dense_27b_config()
    config["text_config"]["hidden_size"] = 4096

    assert qwen36_affine_q8_g64_speculative_block_reason(config) is None


def test_quantization_config_fallback_is_classified():
    config = _qwen36_dense_27b_config()
    config["quantization_config"] = config.pop("quantization")

    assert qwen36_affine_q8_g64_speculative_block_reason(config) is not None


def test_path_predicate_reads_only_config_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen36_dense_27b_config()))

    assert speculative_verification_block_reason_from_path(tmp_path) is not None


@pytest.mark.parametrize("content", ["{", "[]"])
def test_path_predicate_does_not_classify_malformed_configs(tmp_path, content):
    (tmp_path / "config.json").write_text(content)

    assert speculative_verification_block_reason_from_path(tmp_path) is None


def test_path_predicate_does_not_touch_missing_weights(tmp_path):
    # The production predicate is deliberately usable before any safetensors
    # or MLX object is opened.
    (tmp_path / "config.json").write_text(json.dumps(_qwen36_dense_27b_config(bits=4)))

    assert speculative_verification_block_reason_from_path(tmp_path) is None


def _install_fake_mlx(monkeypatch):
    """Install import-only MLX stand-ins; tests must not load the real package."""
    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    core = types.ModuleType("mlx.core")
    core.get_active_memory = lambda: 0
    core.synchronize = lambda: None
    core.clear_cache = lambda: None
    nn = types.ModuleType("mlx.nn")
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda _: []
    mlx.core = core
    mlx.nn = nn
    mlx.utils = utils
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    return core


def _load_production_module(monkeypatch, *, module_name: str, relative_path: str):
    """Execute the checked-in module under a unique name with fake MLX only."""
    root = Path(__file__).resolve().parents[1]
    source_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_native_mtp_admission_disables_the_known_q8_lane(tmp_path, monkeypatch):
    """Exercise model_loading's real gate without importing MLX or a model."""
    _install_fake_mlx(monkeypatch)
    mtp_patch = MagicMock(
        apply_mlx_lm_mtp_patch=MagicMock(return_value=True),
        set_mtp_active=MagicMock(),
        set_mtp_depth=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "omlx.patches.mlx_lm_mtp", mtp_patch)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.m5_gather_qmm",
        MagicMock(apply_m5_gather_qmm_workaround=MagicMock(return_value=False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.arrays_cache_extract",
        MagicMock(apply_arrays_cache_extract_guard=MagicMock()),
    )
    model_loading = _load_production_module(
        monkeypatch,
        module_name="omlx.utils._model_loading_exactness_cpu_test",
        relative_path="omlx/utils/model_loading.py",
    )
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)

    config = _qwen36_dense_27b_config()
    config["mtp_num_hidden_layers"] = 1
    (tmp_path / "config.json").write_text(json.dumps(config))
    settings = types.SimpleNamespace(mtp_enabled=True)

    model_loading.maybe_apply_pre_load_patches(str(tmp_path), model_settings=settings)

    # The first call clears process-wide state. The last call is the actual
    # production MTP admission decision for this target.
    assert mtp_patch.set_mtp_active.call_args_list == [call(False), call(False)]
    mtp_patch.apply_mlx_lm_mtp_patch.assert_called_once_with()

    config["quantization"]["bits"] = 6
    (tmp_path / "config.json").write_text(json.dumps(config))
    mtp_patch.reset_mock()

    model_loading.maybe_apply_pre_load_patches(str(tmp_path), model_settings=settings)

    assert mtp_patch.set_mtp_active.call_args_list == [call(False), call(True)]


def _install_engine_pool_import_stubs(monkeypatch, engine_instances):
    """Make the real EnginePool load method executable without MLX imports."""
    _install_fake_mlx(monkeypatch)

    class FakeBaseEngine:
        pass

    class FakeBatchedEngine(FakeBaseEngine):
        def __init__(self, **_kwargs):
            self.tokenizer = types.SimpleNamespace(encode=lambda _: [])
            self.attached_drafters = []
            self.started = False
            engine_instances.append(self)

        async def start(self):
            self.started = True

        async def stop(self):
            self.started = False

        def set_vlm_mtp_drafter(self, drafter):
            self.attached_drafters.append(drafter)

    engine = types.ModuleType("omlx.engine")
    engine.BaseEngine = FakeBaseEngine
    engine.BatchedEngine = FakeBatchedEngine
    monkeypatch.setitem(sys.modules, "omlx.engine", engine)

    for name, attr in (
        ("omlx.engine.embedding", "EmbeddingEngine"),
        ("omlx.engine.reranker", "RerankerEngine"),
        ("omlx.engine.sts", "STSEngine"),
        ("omlx.engine.stt", "STTEngine"),
        ("omlx.engine.tts", "TTSEngine"),
        ("omlx.engine.vlm", "VLMBatchedEngine"),
    ):
        module = types.ModuleType(name)
        setattr(module, attr, FakeBatchedEngine)
        monkeypatch.setitem(sys.modules, name, module)

    engine_core = types.ModuleType("omlx.engine_core")
    engine_core.get_mlx_executor = lambda: None
    monkeypatch.setitem(sys.modules, "omlx.engine_core", engine_core)

    exceptions = types.ModuleType("omlx.exceptions")
    for name in (
        "InsufficientMemoryError",
        "ModelBusyError",
        "ModelLoadingError",
        "ModelNotFoundError",
        "ModelTooLargeError",
        "ModelUnavailableError",
    ):
        setattr(exceptions, name, type(name, (RuntimeError,), {}))
    exceptions.DEFAULT_CEILING_ADVICE = ""
    exceptions.describe_ceiling_binding = lambda *_args, **_kwargs: ""
    monkeypatch.setitem(sys.modules, "omlx.exceptions", exceptions)

    discovery = types.ModuleType("omlx.model_discovery")
    discovery.discover_models = lambda *_args, **_kwargs: []
    discovery.format_size = lambda value: str(value)
    monkeypatch.setitem(sys.modules, "omlx.model_discovery", discovery)

    class FakeSchedulerConfig:
        def __init__(self):
            self.hot_cache_max_size = 0
            self.hot_cache_budget = None

    scheduler = types.ModuleType("omlx.scheduler")
    scheduler.SchedulerConfig = FakeSchedulerConfig
    monkeypatch.setitem(sys.modules, "omlx.scheduler", scheduler)

    proc_memory = types.ModuleType("omlx.utils.proc_memory")
    proc_memory.get_phys_footprint = lambda: 0
    monkeypatch.setitem(sys.modules, "omlx.utils.proc_memory", proc_memory)
    return FakeBatchedEngine


def test_external_vlm_mtp_admission_skips_drafter_load_for_known_q8_lane(
    tmp_path, monkeypatch
):
    """Run EnginePool's actual post-load branch with observable CPU-only fakes."""
    engine_instances = []
    _install_engine_pool_import_stubs(monkeypatch, engine_instances)
    drafter = object()
    drafter_loader = MagicMock(return_value=drafter)
    monkeypatch.setitem(
        sys.modules,
        "omlx.speculative.vlm_mtp",
        types.SimpleNamespace(load_vlm_mtp_drafter=drafter_loader),
    )
    engine_pool = _load_production_module(
        monkeypatch,
        module_name="omlx._engine_pool_exactness_cpu_test",
        relative_path="omlx/engine_pool.py",
    )

    (tmp_path / "config.json").write_text(json.dumps(_qwen36_dense_27b_config()))
    entry = engine_pool.EngineEntry(
        model_id="known-q8-target",
        model_path=str(tmp_path),
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    pool = engine_pool.EnginePool()
    pool._entries[entry.model_id] = entry
    settings = types.SimpleNamespace(
        dflash_enabled=False,
        trust_remote_code=False,
        vlm_mtp_enabled=True,
        vlm_mtp_draft_model="fake-drafter",
    )

    asyncio.run(pool._load_engine(entry.model_id, runtime_settings=settings))

    assert len(engine_instances) == 1
    assert engine_instances[0].started is True
    assert engine_instances[0].attached_drafters == []
    drafter_loader.assert_not_called()

    # A near-identical Q6 target proves the fakes can traverse the production
    # load-and-attach branch; it is the Q8 gate, not incomplete test plumbing,
    # that keeps the first target on ordinary decode.
    q6_path = tmp_path / "q6-control"
    q6_path.mkdir()
    q6_config = _qwen36_dense_27b_config(bits=6)
    (q6_path / "config.json").write_text(json.dumps(q6_config))
    q6_entry = engine_pool.EngineEntry(
        model_id="q6-control",
        model_path=str(q6_path),
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    q6_pool = engine_pool.EnginePool()
    q6_pool._entries[q6_entry.model_id] = q6_entry

    asyncio.run(q6_pool._load_engine(q6_entry.model_id, runtime_settings=settings))

    assert len(engine_instances) == 2
    assert engine_instances[1].attached_drafters == [drafter]
    drafter_loader.assert_called_once_with("fake-drafter")
