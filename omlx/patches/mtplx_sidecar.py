# SPDX-License-Identifier: Apache-2.0
"""Loader support for MTPLX-style MTP sidecar checkpoints.

Some MTPLX checkpoints keep the native MTP head in a separate file declared by
``config.json``::

    "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"}

The stock mlx-lm text loader only globs ``model*.safetensors`` so it never sees
that sidecar.  The mlx-vlm loader globs every ``*.safetensors`` and therefore
*does* see it, but the raw sidecar keys are rooted at ``mtp.*`` while the VLM
model tree expects ``language_model.mtp.*``.

This patch wraps the upstream load_model functions narrowly:

* mlx-lm: append the declared sidecar to the weight glob only when Native MTP is
  active for the current load.
* mlx-vlm: filter the sidecar out when Native MTP is inactive; when active, keep
  it and prefix raw ``mtp.*`` keys to ``language_model.mtp.*``.

The wrappers are process-wide and idempotent; the per-call behavior is driven by
``mlx_lm_mtp.is_mtp_active()`` and the model directory's config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_mtplx_sidecar_patch() -> bool:
    """Install MTPLX sidecar-aware wrappers for mlx-lm / mlx-vlm loaders."""
    global _PATCHED
    if _PATCHED:
        return True

    applied = False
    try:
        import mlx_lm.utils as lm_utils
    except Exception as e:
        logger.debug("mlx-lm utils not importable for MTPLX sidecar patch: %s", e)
    else:
        _patch_load_model(lm_utils, mode="lm")
        applied = True

    try:
        import mlx_vlm.utils as vlm_utils
    except Exception as e:
        logger.debug("mlx-vlm utils not importable for MTPLX sidecar patch: %s", e)
    else:
        _patch_load_model(vlm_utils, mode="vlm")
        applied = True

    _PATCHED = applied
    if applied:
        logger.info("MTPLX MTP sidecar loader patch applied")
    return applied


def get_mtplx_mtp_sidecar(model_dir: str | Path) -> Path | None:
    """Return the declared MTP sidecar path if config declares one and it exists."""
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None

    extra = cfg.get("mlx_lm_extra_tensors") or {}
    mtp_file = extra.get("mtp_file")
    if not mtp_file:
        return None

    sidecar = (model_dir / str(mtp_file)).resolve()
    try:
        sidecar.relative_to(model_dir.resolve())
    except ValueError:
        logger.warning("Ignoring MTPLX mtp_file outside model dir: %s", sidecar)
        return None
    return sidecar if sidecar.exists() else None


def _mtp_active() -> bool:
    try:
        from .mlx_lm_mtp import is_mtp_active

        return bool(is_mtp_active())
    except Exception:
        return False


def _patch_load_model(utils_mod: Any, *, mode: str) -> None:
    marker = f"_omlx_mtplx_sidecar_patched_{mode}"
    if getattr(utils_mod, marker, False):
        return

    original_load_model = utils_mod.load_model

    def wrapped_load_model(model_path, *args, **kwargs):
        model_dir = Path(model_path)
        sidecar = get_mtplx_mtp_sidecar(model_dir)
        if sidecar is None:
            return original_load_model(model_path, *args, **kwargs)

        active = _mtp_active()
        original_glob = utils_mod.glob.glob
        original_mx_load: Callable[..., Any] = utils_mod.mx.load
        original_load_config = getattr(utils_mod, "load_config", None)

        def patched_glob(pattern, *g_args, **g_kwargs):
            paths = list(original_glob(pattern, *g_args, **g_kwargs))
            try:
                pat = Path(pattern)
            except TypeError:
                return paths

            sidecar_s = str(sidecar)
            if mode == "lm" and pat.name == "model*.safetensors":
                if active and sidecar_s not in paths:
                    paths.append(sidecar_s)
            elif mode == "vlm" and pat.name == "*.safetensors":
                if active:
                    if sidecar_s not in paths:
                        paths.append(sidecar_s)
                else:
                    paths = [p for p in paths if str(Path(p).resolve()) != sidecar_s]
            return paths

        def patched_mx_load(file, *l_args, **l_kwargs):
            weights = original_mx_load(file, *l_args, **l_kwargs)
            try:
                is_sidecar = Path(file).resolve() == sidecar
            except TypeError:
                is_sidecar = False
            if not is_sidecar:
                return weights
            return {_prefix_mtp_key(k): v for k, v in weights.items()}

        def patched_load_config(*c_args, **c_kwargs):
            cfg = original_load_config(*c_args, **c_kwargs)
            if active:
                _augment_config_with_mtp_quantization(cfg, sidecar)
            return cfg

        utils_mod.glob.glob = patched_glob
        utils_mod.mx.load = patched_mx_load
        if original_load_config is not None:
            utils_mod.load_config = patched_load_config
        try:
            return original_load_model(model_path, *args, **kwargs)
        finally:
            utils_mod.glob.glob = original_glob
            utils_mod.mx.load = original_mx_load
            if original_load_config is not None:
                utils_mod.load_config = original_load_config

    utils_mod.load_model = wrapped_load_model
    setattr(utils_mod, marker, True)


def _augment_config_with_mtp_quantization(config: dict, sidecar: Path) -> None:
    """Add fine-grained quantization entries for prequantized MTP sidecars.

    MTPLX stores the MTP head with its own quantization policy (currently
    group_size=32 INT4) while the trunk config may use a different default
    (for example group_size=64). mlx-lm/mlx-vlm decide how to instantiate each
    QuantizedLinear from ``config["quantization"]`` before loading weights, so
    the sidecar's packed shapes only line up if we surface per-layer MTP entries.
    """
    mtp_q = config.get("mtplx_mtp_quantization") or {}
    params = {
        "group_size": int(mtp_q.get("group_size", 32) or 32),
        "bits": int(mtp_q.get("bits", 4) or 4),
        "mode": mtp_q.get("mode", "affine") or "affine",
    }

    try:
        from safetensors import safe_open

        with safe_open(str(sidecar), framework="numpy") as f:  # type: ignore[arg-type]
            keys = set(f.keys())
    except Exception as e:
        logger.debug("Could not inspect MTPLX sidecar quantization keys: %s", e)
        return

    quant = config.setdefault("quantization", {})
    for key in keys:
        if not key.endswith(".weight"):
            continue
        stem = key[: -len(".weight")]
        # Only prequantized linears carry scales. BF16 tensors such as mtp.fc
        # and RMSNorm weights intentionally stay out of the quantization map.
        if f"{stem}.scales" not in keys:
            continue
        quant.setdefault(_prefix_mtp_key(stem), params)


def _prefix_mtp_key(key: str) -> str:
    if key.startswith("language_model."):
        return key
    if key.startswith("mtp."):
        return "language_model." + key
    return key
