# SPDX-License-Identifier: Apache-2.0
"""Cache type handlers for PoolingCache and BatchPoolingCache.

DeepSeek V4 uses these caches for the compressed (sliding-window) attention
path. They are not sliceable on a per-token basis because the pool is
compressed in fixed ``ratio``-sized windows. Both handlers expose a full
state round-trip (extract → reconstruct) so SSD eviction and recovery work,
but ``supports_block_slicing = False`` keeps the prefix cache from trying
to dedup partial windows.
"""

from __future__ import annotations

import logging
from typing import Any

from omlx.cache.type_handlers import (
    CacheStateAxisInfo,
    CacheType,
    CacheTypeHandler,
)

logger = logging.getLogger(__name__)


class PoolingCacheHandler(CacheTypeHandler):
    """Handler for ``mlx_lm.models.cache.PoolingCache`` (single-sequence)."""

    @property
    def cache_type(self) -> CacheType:
        return CacheType.POOLING_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        # Compressed pool — partial slicing is not meaningful.
        return False

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        buf_kv, buf_gate, pooled = cache_obj.state
        return {
            "buf_kv": buf_kv,
            "buf_gate": buf_gate,
            "pooled": pooled,
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        pooled = state.get("pooled")
        if pooled is not None and hasattr(pooled, "shape") and len(pooled.shape) >= 2:
            return int(pooled.shape[1])
        return 0

    def slice_state(
        self,
        state: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any] | None:
        # PoolingCache is not block-sliceable; return the full state with a
        # marker so the storage layer treats it as opaque.
        return {**state, "is_full_state": True}

    def concatenate_states(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Per-block concatenation is not supported. Use the most recent
        # state — same convention as RotatingKVCacheHandler.
        return states[-1] if states else {}

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            from mlx_lm.models.cache import PoolingCache
        except ImportError:
            logger.error("mlx_lm.models.cache.PoolingCache unavailable for reconstruct")
            return None

        ratio = meta_state if isinstance(meta_state, int) else 1
        cache = PoolingCache(ratio=ratio)
        cache.state = (
            state.get("buf_kv"),
            state.get("buf_gate"),
            state.get("pooled"),
        )
        return cache

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        # PoolingCache.state = (buf_kv, buf_gate, pooled).
        # buf_kv / buf_gate are remainder windows of shape (B, ratio, D);
        # pooled is the accumulated compressed sequence (B, P, D). The
        # quantization at ``ratio`` makes per-token slicing unsafe — keep
        # all three elements non-sliceable so omlx core takes the
        # last-block-only / boundary-snapshot path.
        return (
            CacheStateAxisInfo(name="buf_kv", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="buf_gate", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="pooled", sequence_axis=1, sliceable=False),
        )

    def deserialize_state(
        self,
        elements: tuple[Any, ...],
        meta_state: Any | None = None,
    ) -> Any:
        """Reconstruct PoolingCache from a 3-tuple state directly.

        omlx core dispatches handlers via ``deserialize_state`` instead of
        the legacy keys/values dict so 3-tuple state survives without
        getting truncated by the default 2-tuple mapping.
        """
        if not isinstance(elements, (list, tuple)):
            logger.error(
                "PoolingCache deserialize: expected tuple, got %s",
                type(elements).__name__,
            )
            return None
        # Tolerate length-2 input (legacy V2-truncated state); fill the
        # missing pooled with None so reconstruct doesn't crash.
        if len(elements) == 2:
            buf_kv, buf_gate = elements
            pooled = None
        elif len(elements) == 3:
            buf_kv, buf_gate, pooled = elements
        else:
            logger.error(
                "PoolingCache deserialize: expected 2 or 3 elements, got %d",
                len(elements),
            )
            return None
        return self.reconstruct_cache(
            {"buf_kv": buf_kv, "buf_gate": buf_gate, "pooled": pooled},
            meta_state,
        )

    def _get_state_keys(self) -> tuple[str, ...]:
        return ("buf_kv", "buf_gate", "pooled")

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("ratio",)


class BatchPoolingCacheHandler(CacheTypeHandler):
    """Handler for ``mlx_lm.models.cache.BatchPoolingCache``.

    BatchPoolingCache state is the same 3-tuple as PoolingCache but kept
    untrimmed; meta_state is a 4-tuple
    ``(ratio, remainder, _pool_lengths, _processed)``.
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.BATCH_POOLING_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        return False

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        buf_kv, buf_gate, pooled = cache_obj.state
        return {
            "buf_kv": buf_kv,
            "buf_gate": buf_gate,
            "pooled": pooled,
            "cache_type": self.cache_type.value,
            "is_full_state": True,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        pooled = state.get("pooled")
        if pooled is not None and hasattr(pooled, "shape") and len(pooled.shape) >= 2:
            return int(pooled.shape[1])
        return 0

    def slice_state(
        self,
        state: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any] | None:
        return {**state, "is_full_state": True}

    def concatenate_states(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return states[-1] if states else {}

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            from mlx_lm.models.cache import BatchPoolingCache
        except ImportError:
            logger.error(
                "mlx_lm.models.cache.BatchPoolingCache unavailable for reconstruct"
            )
            return None

        if not isinstance(meta_state, tuple) or len(meta_state) != 4:
            logger.error(
                "BatchPoolingCache reconstruct expects 4-tuple meta_state "
                "(ratio, remainder, pool_lengths, processed); got %r",
                type(meta_state).__name__,
            )
            return None

        ratio, remainder, pool_lengths, processed = meta_state
        batch_size = len(remainder)
        cache = BatchPoolingCache(ratio=ratio, left_padding=[0] * batch_size)
        cache.state = (
            state.get("buf_kv"),
            state.get("buf_gate"),
            state.get("pooled"),
        )
        cache.meta_state = (ratio, list(remainder), list(pool_lengths), list(processed))
        return cache

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        return (
            CacheStateAxisInfo(name="buf_kv", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="buf_gate", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="pooled", sequence_axis=1, sliceable=False),
        )

    def deserialize_state(
        self,
        elements: tuple[Any, ...],
        meta_state: Any | None = None,
    ) -> Any:
        if not isinstance(elements, (list, tuple)):
            logger.error(
                "BatchPoolingCache deserialize: expected tuple, got %s",
                type(elements).__name__,
            )
            return None
        if len(elements) == 2:
            buf_kv, buf_gate = elements
            pooled = None
        elif len(elements) == 3:
            buf_kv, buf_gate, pooled = elements
        else:
            logger.error(
                "BatchPoolingCache deserialize: expected 2 or 3 elements, got %d",
                len(elements),
            )
            return None
        return self.reconstruct_cache(
            {"buf_kv": buf_kv, "buf_gate": buf_gate, "pooled": pooled},
            meta_state,
        )

    def _get_state_keys(self) -> tuple[str, ...]:
        return ("buf_kv", "buf_gate", "pooled")

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("ratio", "remainder", "pool_lengths", "processed")


# The 8 V4Cache state field names, in the same order as V4Cache.state /
# CacheCheckpoint._SLOTS (sans offset, which travels in meta_state).
_V4_STATE_FIELDS: tuple[str, ...] = (
    "keys",
    "values",
    "compressed",
    "compressor_kv_state",
    "compressor_score_state",
    "indexer_compressed",
    "indexer_compressor_kv_state",
    "indexer_compressor_score_state",
)


class V4CacheHandler(CacheTypeHandler):
    """Handler for DeepSeek V4 ``V4Cache`` (per-layer sliding-window + pool).

    V4Cache carries 8 mx.array fields beyond a vanilla KVCache: the
    sliding-window keys/values plus six compressed-stream pool arrays
    (the attention compressor's running buffer + state machine, plus a
    separate set for the CSA indexer's internal compressor). All eight
    must round-trip through SSD eviction / boundary snapshots, otherwise
    decode after restore desyncs against the JANGTQ kernels.

    ``supports_block_slicing = False``: the windowed-causal mask and
    the ratio-quantized compressed pool both make partial slicing
    meaningless — omlx core takes the last-block-only path instead.
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.V4_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        # Window + compressed pool — partial slicing is not meaningful.
        return False

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        # Read through ``cache_obj.state`` (the 8-tuple), not via getattr —
        # keeps the cache-object abstraction clean and ensures handler
        # behavior tracks the V4Cache.state contract.
        elements = cache_obj.state
        out: dict[str, Any] = dict(zip(_V4_STATE_FIELDS, elements))
        out["cache_type"] = self.cache_type.value
        out["is_full_state"] = True
        return out

    def get_seq_len(self, state: dict[str, Any]) -> int:
        keys = state.get("keys")
        if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
            # V4 keys are shape (B, n_kv=1, T, head_dim) — sequence on axis 2.
            return int(keys.shape[2])
        return 0

    def slice_state(
        self,
        state: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any] | None:
        # Non-sliceable; return the full state with the opaque marker
        # so the storage layer treats it as an atomic boundary snapshot.
        return {**state, "is_full_state": True}

    def concatenate_states(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Per-block concatenation is not supported — use the most recent
        # state, matching PoolingCacheHandler / RotatingKVCacheHandler.
        return states[-1] if states else {}

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            from mlx_lm.models.deepseek_v4 import V4Cache
        except ImportError:
            logger.error(
                "mlx_lm.models.deepseek_v4.V4Cache unavailable for reconstruct "
                "(did apply_deepseek_v4_patch run first?)"
            )
            return None

        if not isinstance(meta_state, (list, tuple)) or len(meta_state) not in (3, 4):
            logger.error(
                "V4Cache reconstruct expects a 3- or 4-tuple meta_state "
                "(layer_idx, compress_ratio, offset[, window_size]); got %r",
                type(meta_state).__name__,
            )
            return None

        layer_idx = meta_state[0]
        compress_ratio = meta_state[1]
        offset = meta_state[2]
        # window_size is only optional for forward-compat with legacy
        # 3-tuple meta_state from before this handler existed. Fall back to
        # the V4 default (per ModelArgs.sliding_window in deepseek_v4_model).
        window_size = meta_state[3] if len(meta_state) >= 4 else 128

        cache = V4Cache.from_state(
            layer_idx=layer_idx,
            compress_ratio=compress_ratio,
            offset=offset,
            window_size=window_size,
        )
        # Assemble the 8-tuple from the state dict; missing keys → None
        # (preserves the "unset until first compressor step" invariant).
        cache.state = tuple(state.get(name) for name in _V4_STATE_FIELDS)
        return cache

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        # All entries marked non-sliceable: the window + compressed-pool
        # combo doesn't support partial per-token slicing, so omlx core
        # falls back to last-block-only / boundary-snapshot storage.
        return (
            # keys / values: shape (B, n_kv=1, T, head_dim), seq on axis 2.
            CacheStateAxisInfo(name="keys", sequence_axis=2, sliceable=False),
            CacheStateAxisInfo(name="values", sequence_axis=2, sliceable=False),
            # compressed: shape (B, max_seq_len/ratio, head_dim), seq on axis 1.
            CacheStateAxisInfo(name="compressed", sequence_axis=1, sliceable=False),
            # State-machine arrays (compressor + indexer pool state) — these
            # carry partial-window accumulation, not sequence-indexed data.
            CacheStateAxisInfo(name="compressor_kv_state", sequence_axis=None, sliceable=False),
            CacheStateAxisInfo(name="compressor_score_state", sequence_axis=None, sliceable=False),
            # indexer_compressed mirrors compressed's layout.
            CacheStateAxisInfo(name="indexer_compressed", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="indexer_compressor_kv_state", sequence_axis=None, sliceable=False),
            CacheStateAxisInfo(name="indexer_compressor_score_state", sequence_axis=None, sliceable=False),
        )

    def deserialize_state(
        self,
        elements: tuple[Any, ...],
        meta_state: Any | None = None,
    ) -> Any:
        """Reconstruct V4Cache from an 8-tuple state.

        Tolerates length-2 input (legacy keys/values-only) for forward
        compatibility with caches serialized before the 8-field state
        contract — missing compressor fields fill in as None.
        """
        if not isinstance(elements, (list, tuple)):
            logger.error(
                "V4Cache deserialize: expected tuple, got %s",
                type(elements).__name__,
            )
            return None

        if len(elements) == 2:
            state_dict: dict[str, Any] = {
                "keys": elements[0],
                "values": elements[1],
            }
            for name in _V4_STATE_FIELDS[2:]:
                state_dict[name] = None
        elif len(elements) == 8:
            state_dict = dict(zip(_V4_STATE_FIELDS, elements))
        else:
            logger.error(
                "V4Cache deserialize: expected 2 or 8 elements, got %d",
                len(elements),
            )
            return None

        return self.reconstruct_cache(state_dict, meta_state)

    def _get_state_keys(self) -> tuple[str, ...]:
        return _V4_STATE_FIELDS

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("layer_idx", "compress_ratio", "offset", "window_size")


class BatchV4CacheHandler(CacheTypeHandler):
    """Handler for DeepSeek V4 ``BatchV4Cache``.

    BatchV4Cache shares V4Cache's 8 state arrays but keeps the leading
    batch axis populated across concurrent requests. ``meta_state`` is a
    7-tuple ``(layer_idx, compress_ratio, offset, window_size,
    left_padding, lengths, processed)`` — same scalar quadruple as
    V4Cache plus three per-row Python lists.

    Like V4CacheHandler, block slicing is disabled: the windowed-causal
    mask + ratio-quantized compressed pool make per-token slicing
    meaningless. The boundary-snapshot path is used end-to-end.
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.BATCH_V4_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        return False

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        elements = cache_obj.state
        out: dict[str, Any] = dict(zip(_V4_STATE_FIELDS, elements))
        out["cache_type"] = self.cache_type.value
        out["is_full_state"] = True
        return out

    def get_seq_len(self, state: dict[str, Any]) -> int:
        keys = state.get("keys")
        if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
            # BatchV4Cache keys are shape (B, n_kv=1, T, head_dim) — seq on axis 2.
            return int(keys.shape[2])
        return 0

    def slice_state(
        self,
        state: dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any] | None:
        # Non-sliceable; opaque boundary snapshot.
        return {**state, "is_full_state": True}

    def concatenate_states(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return states[-1] if states else {}

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            from mlx_lm.models.deepseek_v4 import BatchV4Cache
        except ImportError:
            logger.error(
                "mlx_lm.models.deepseek_v4.BatchV4Cache unavailable for "
                "reconstruct (did apply_deepseek_v4_patch run first?)"
            )
            return None

        if not isinstance(meta_state, (list, tuple)) or len(meta_state) != 7:
            logger.error(
                "BatchV4Cache reconstruct expects a 7-tuple meta_state "
                "(layer_idx, compress_ratio, offset, window_size, "
                "left_padding, lengths, processed); got %r",
                type(meta_state).__name__,
            )
            return None

        (
            layer_idx,
            compress_ratio,
            offset,
            window_size,
            left_padding,
            lengths,
            processed,
        ) = meta_state

        cache = BatchV4Cache(
            layer_idx=layer_idx,
            compress_ratio=compress_ratio,
            window_size=window_size,
            left_padding=list(left_padding),
        )
        cache.offset = int(offset)
        cache._lengths = list(lengths)
        cache._processed = list(processed)
        # Assemble the 8-tuple from the state dict; missing keys → None
        # (preserves the "unset until first compressor step" invariant).
        cache.state = tuple(state.get(name) for name in _V4_STATE_FIELDS)
        return cache

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        # Same 8 entries as V4CacheHandler — only the leading batch axis
        # is populated, the per-array seq-axis layout is identical.
        return (
            CacheStateAxisInfo(name="keys", sequence_axis=2, sliceable=False),
            CacheStateAxisInfo(name="values", sequence_axis=2, sliceable=False),
            CacheStateAxisInfo(name="compressed", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="compressor_kv_state", sequence_axis=None, sliceable=False),
            CacheStateAxisInfo(name="compressor_score_state", sequence_axis=None, sliceable=False),
            CacheStateAxisInfo(name="indexer_compressed", sequence_axis=1, sliceable=False),
            CacheStateAxisInfo(name="indexer_compressor_kv_state", sequence_axis=None, sliceable=False),
            CacheStateAxisInfo(name="indexer_compressor_score_state", sequence_axis=None, sliceable=False),
        )

    def deserialize_state(
        self,
        elements: tuple[Any, ...],
        meta_state: Any | None = None,
    ) -> Any:
        """Reconstruct BatchV4Cache from an 8-tuple state.

        Same length-2 / length-8 tolerance as V4CacheHandler.
        """
        if not isinstance(elements, (list, tuple)):
            logger.error(
                "BatchV4Cache deserialize: expected tuple, got %s",
                type(elements).__name__,
            )
            return None

        if len(elements) == 2:
            state_dict: dict[str, Any] = {
                "keys": elements[0],
                "values": elements[1],
            }
            for name in _V4_STATE_FIELDS[2:]:
                state_dict[name] = None
        elif len(elements) == 8:
            state_dict = dict(zip(_V4_STATE_FIELDS, elements))
        else:
            logger.error(
                "BatchV4Cache deserialize: expected 2 or 8 elements, got %d",
                len(elements),
            )
            return None

        return self.reconstruct_cache(state_dict, meta_state)

    def _get_state_keys(self) -> tuple[str, ...]:
        return _V4_STATE_FIELDS

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return (
            "layer_idx",
            "compress_ratio",
            "offset",
            "window_size",
            "left_padding",
            "lengths",
            "processed",
        )
