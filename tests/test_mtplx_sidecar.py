# SPDX-License-Identifier: Apache-2.0
"""Tests for MTPLX MTP sidecar checkpoint support."""

from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import save_file

from omlx.admin.routes import _model_has_mtp_weight_tensors
from omlx.patches.mtplx_sidecar import (
    _augment_config_with_mtp_quantization,
    _prefix_mtp_key,
    get_mtplx_mtp_sidecar,
)


def test_declared_mtplx_sidecar_counts_as_mtp_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"}})
    )
    save_file({"mtp.fc.weight": np.zeros((1, 1), dtype=np.float32)}, tmp_path / "mtp.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))

    assert get_mtplx_mtp_sidecar(tmp_path) == (tmp_path / "mtp.safetensors").resolve()
    assert _model_has_mtp_weight_tensors(tmp_path) is True


def test_missing_or_non_mtp_sidecar_does_not_count(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"}})
    )
    save_file({"model.layers.0.weight": np.zeros((1, 1), dtype=np.float32)}, tmp_path / "mtp.safetensors")

    assert _model_has_mtp_weight_tensors(tmp_path) is False


def test_mtplx_sidecar_prefixes_mtp_under_language_model():
    assert _prefix_mtp_key("mtp.fc.weight") == "language_model.mtp.fc.weight"
    assert _prefix_mtp_key("language_model.mtp.fc.weight") == "language_model.mtp.fc.weight"
    assert _prefix_mtp_key("other.weight") == "other.weight"


def test_mtplx_sidecar_adds_fine_grained_quantization(tmp_path):
    sidecar = tmp_path / "mtp.safetensors"
    save_file(
        {
            "mtp.layers.0.self_attn.q_proj.weight": np.zeros((1, 1), dtype=np.uint32),
            "mtp.layers.0.self_attn.q_proj.scales": np.zeros((1, 1), dtype=np.float32),
            "mtp.fc.weight": np.zeros((1, 1), dtype=np.float32),
        },
        sidecar,
    )
    config = {"mtplx_mtp_quantization": {"bits": 4, "group_size": 32, "mode": "affine"}}

    _augment_config_with_mtp_quantization(config, sidecar)

    assert config["quantization"] == {
        "language_model.mtp.layers.0.self_attn.q_proj": {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
        }
    }
