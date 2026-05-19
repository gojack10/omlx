"""DeepSeek V4 Flash MLX implementation.

Architecture port of DeepSeek V4 (43-layer MoE with HC residuals, sliding-window +
compressed-stream KV cache, sparse-topk indexer on compress_ratio=4 layers).
Math ported directly from `~/.omlx/models/DeepSeek-V4-Flash/inference/model.py`
(DeepSeek's PyTorch reference). Targets oMLX-bundled mlx_lm; loads JANGTQ-quant
checkpoint at `~/.omlx/models/DeepSeek-V4-Flash-JANGTQ/`.

Stage 1 scope (this file):
  - Architecture, Attention with grouped wo_a/wo_b + attn_sink, HC residuals,
    MoE with hash routing on first 3 layers + sqrtsoftplus elsewhere, sanitize
    for the JANGTQ checkpoint (drop mtp, dequant wo_a to plain bf16, pass through
    8-bit affine tensors for attention/shared/indexer/embed/head/compressor).
  - Routed-expert MoE dispatch via JANGTQ 2-bit MXTQ kernel: per-expert
    on-the-fly decode (W_rotated[i, k] = tq_norms[i] * codebook_N[unpack(tq_packed)[i, k]])
    plus randomized-Hadamard rotation of inputs (signs * x then Walsh-Hadamard
    over the input dim, scaled by 1/sqrt(N)). Codebook + sign vectors are
    ingested via the `jangtq_runtime.safetensors` sidecar which must be
    discoverable by mlx_lm — symlink it as `model-jangtq-runtime.safetensors`
    in the model dir so the standard `model*.safetensors` glob picks it up.
  - CSA Compressor and sparse Indexer constructed but not yet invoked
    (forward path falls back to dense sliding-window attention). Adding them
    is an obvious next step once Compressor+Indexer math is ported.

Bit-for-bit parity vs PyTorch reference: deferred to a later session.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_causal_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import _BaseCache
from mlx_lm.models.rope_utils import initialize_rope


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"

    # Embedding / output
    vocab_size: int = 129280
    hidden_size: int = 4096
    tie_word_embeddings: bool = False

    # Layers
    num_hidden_layers: int = 43
    num_hash_layers: int = 3   # first N layers' MoE Gate uses tid2eid lookup

    # Attention (MLA shape — V4 changed from V32)
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512                 # unified; last `qk_rope_head_dim` are rope-rotated
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    attention_bias: bool = False

    # Indexer (V4 still has one — separate from cache, only on compress_ratio==4 layers)
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512

    # MoE (256 routed + 1 shared, 6 active, sqrtsoftplus)
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True

    # HC (Hyper-Connections, the residual replacement scheme)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # CSA (Compressed Sparse Attention) — per-layer compression ratios
    compress_rope_theta: float = 160000.0
    compress_ratios: list[int] = field(default_factory=lambda: [0] * 44)
    sliding_window: int = 128
    # Phase 1 cap for the per-layer compressed-stream KV buffer.
    # max_compressed_positions = compressed_max_seq_len // compress_ratio.
    # Must accommodate the longest realistic prompt + generation length
    # (32K tokens here). Phase 2 work decouples buffer sizing from this cap.
    compressed_max_seq_len: int = 32768

    # mHC swiglu clip
    swiglu_limit: float = 10.0

    # RoPE / context (YaRN)
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 10000.0
    rope_parameters: Optional[dict] = None

    # Norm
    rms_norm_eps: float = 1e-6

    # MTP (dropped at sanitize time per jang_config.json drop_mtp=true)
    num_nextn_predict_layers: int = 1

    # MTP-native speculative drafter — Option 3 hybrid bolt-on. When True,
    # construct an MTPBlock alongside the main model and read its weights
    # from `mtp_source_dir` (non-JANGTQ Flash). Pre-dequant to BF16 at load
    # time (~12 GB resident; 256 routed experts dominate). See the
    # standalone loader `_load_mtp_weights.py` for the dequant math.
    load_mtp: bool = False
    mtp_source_dir: str = "/Users/jack/.omlx/models/DeepSeek-V4-Flash"

    # Quant metadata (informational)
    quantization: Optional[dict] = None
    routed_expert_bits: int = 2
    mxtq_seed: int = 42
    group_size: int = 32

    # Hidden activation
    hidden_act: str = "silu"


# ---------------------------------------------------------------------------
# Cache — sliding window + (when present) compressed-stream KV
# ---------------------------------------------------------------------------


# Module-level slot tuples shared between V4Cache.state and CacheCheckpoint to
# keep speculative-decode rollback and (de)serialization in lockstep — if one
# adds a field the other gets it for free.
_V4_CACHE_STATE_ARRAY_SLOTS: tuple[str, ...] = (
    "keys",
    "values",
    "compressed",
    "compressor_kv_state",
    "compressor_score_state",
    "indexer_compressed",
    "indexer_compressor_kv_state",
    "indexer_compressor_score_state",
)


class V4Cache(_BaseCache):
    """Per-layer V4 KV cache.

    For Stage 1: sliding-window-only growable KV. The compressed-stream
    component documented in the V4 paper is allocated lazily but unused
    until CSACompressor.__call__ is filled in.

    Stores `keys` (= `values` in V4; KV is shared, single n_kv_head=1, head_dim=512)
    as a growing buffer. When offset > sliding_window during decode, callers
    should slice to the last `sliding_window` tokens. (Stage 1 prefill of
    short sequences never trips this.)
    """

    def __init__(
        self,
        args: Optional[ModelArgs] = None,
        layer_idx: Optional[int] = None,
        *,
        compress_ratio: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        """Construct a V4Cache.

        Two call shapes are supported:

        * Normal model construction: ``V4Cache(args, layer_idx)`` — pulls
          ``compress_ratio`` / ``window_size`` from ``args``.
        * Deserialization construction: ``V4Cache(args=None, layer_idx=L,
          compress_ratio=R, window_size=W)`` — used by the omlx cache
          handler's ``reconstruct_cache`` path which only has the meta_state
          scalars, not a full ``ModelArgs``. See
          ``V4Cache.from_state`` for the convenience classmethod.
        """
        super().__init__()
        if args is not None:
            if layer_idx is None:
                raise TypeError("V4Cache(args=...) requires layer_idx")
            self.args = args
            self.layer_idx = layer_idx
            self.compress_ratio = args.compress_ratios[layer_idx]
            self.window_size = args.sliding_window
        else:
            if layer_idx is None or compress_ratio is None or window_size is None:
                raise TypeError(
                    "V4Cache(args=None) requires explicit "
                    "layer_idx, compress_ratio, and window_size"
                )
            self.args = None
            self.layer_idx = layer_idx
            self.compress_ratio = compress_ratio
            self.window_size = window_size
        # V4 keys == values (single shared latent), shape (B, n_kv=1, T, head_dim=512)
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        # Compressed-stream component (CSA / HCA layers only): the running kv_cache
        # buffer that the attention's Compressor writes to. (B, max_seq_len/ratio, head_dim).
        self.compressed: Optional[mx.array] = None
        # Per-layer Compressor state (CSA / HCA layers only): partial-window pool
        # state machine carried across decode steps. Shape per ref model.py:303-304:
        #   (B, coff*ratio, coff*head_dim)  with coff = 1 + (ratio==4).
        self.compressor_kv_state: Optional[mx.array] = None
        self.compressor_score_state: Optional[mx.array] = None
        # Indexer's internal compressor (CSA layers only — compress_ratio==4):
        # separate kv_cache + state so writes don't collide with the attention's.
        self.indexer_compressed: Optional[mx.array] = None
        self.indexer_compressor_kv_state: Optional[mx.array] = None
        self.indexer_compressor_score_state: Optional[mx.array] = None
        self.offset = 0

    @classmethod
    def from_state(
        cls,
        layer_idx: int,
        compress_ratio: int,
        offset: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> "V4Cache":
        """Construct an empty V4Cache from serialized meta_state scalars.

        Used by the omlx cache handler to rebuild a V4Cache during cache
        restore / SSD eviction recovery, where the original ``ModelArgs``
        isn't available. Accepts the meta_state quadruple positionally
        (``layer_idx, compress_ratio, offset, window_size``) so callers
        can splat ``*cache.meta_state`` directly; ``offset`` is applied to
        the new cache (call sites that prefer to set offset via the
        ``state``/``meta_state`` setters can leave it None).

        ``window_size`` is required for any reconstruction; callers
        recovering pre-window_size meta_state should supply the model's
        default sliding_window explicitly.
        """
        if window_size is None:
            raise TypeError("V4Cache.from_state requires window_size")
        cache = cls(
            args=None,
            layer_idx=layer_idx,
            compress_ratio=compress_ratio,
            window_size=window_size,
        )
        if offset is not None:
            cache.offset = offset
        return cache

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append new keys/values; return the slice that the attention call
        should attend against.

        Prefill (S > 1): return the full grown cache so the (S, S)
        windowed-causal mask broadcasts cleanly against scores
        (B, nh, S, offset). The mask's `linds < rinds + window_size` term
        enforces the sliding window on the query axis. This mirrors the
        reference (`inference/model.py::Attention.forward`, start_pos == 0
        path) which attends q against the full local kv, not the bounded
        kv_cache.

        Decode (S == 1): slice to the rightmost `window_size` tokens. Mask
        is None at S==1 (`create_attention_mask` returns None for N==1), so
        the slice both bounds memory and produces the correct attention
        pattern for a single-token query.
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = self.keys.shape[2]
        new_S = keys.shape[2]
        if new_S == 1 and self.offset > self.window_size:
            return (
                self.keys[..., -self.window_size:, :],
                self.values[..., -self.window_size:, :],
            )
        return self.keys, self.values

    @property
    def state(self):
        # 8-tuple matching _V4_CACHE_STATE_ARRAY_SLOTS. Entries are None when
        # unset (e.g. compressed-stream arrays for layers that haven't run
        # CSACompressor yet); preserve None — do not substitute zeros.
        # ``offset`` is carried separately via ``meta_state`` so the array
        # slots stay dtype-homogeneous (all mx.array or None).
        return tuple(getattr(self, name) for name in _V4_CACHE_STATE_ARRAY_SLOTS)

    @state.setter
    def state(self, value):
        # Tolerate length-2 (legacy keys/values-only) and length-8 inputs.
        # Length-2 leaves the 6 compressor fields at their current value;
        # callers that need them cleared should construct via ``from_state``
        # first (which initializes them to None).
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"V4Cache.state must be assigned a tuple, got {type(value).__name__}"
            )
        if len(value) == 2:
            self.keys, self.values = value
        elif len(value) == 8:
            for name, v in zip(_V4_CACHE_STATE_ARRAY_SLOTS, value):
                setattr(self, name, v)
        else:
            raise ValueError(
                f"V4Cache.state expects a 2- or 8-tuple, got length {len(value)}"
            )

    @property
    def meta_state(self):
        # offset is stored here (not in ``state``) so the array tuple stays
        # homogeneous. window_size is carried so the handler's
        # ``from_state`` reconstructor (which has no ``args``) can rebuild
        # the cache identically.
        return (self.layer_idx, self.compress_ratio, self.offset, self.window_size)

    @meta_state.setter
    def meta_state(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"V4Cache.meta_state must be assigned a tuple, got {type(value).__name__}"
            )
        if len(value) == 3:
            # Forward-compat: accept legacy 3-tuple (no window_size).
            self.layer_idx, self.compress_ratio, self.offset = value
        elif len(value) == 4:
            self.layer_idx, self.compress_ratio, self.offset, self.window_size = value
        else:
            raise ValueError(
                f"V4Cache.meta_state expects a 3- or 4-tuple, got length {len(value)}"
            )

    def is_trimmable(self) -> bool:
        return False  # window-only logic; compressed stream not yet wired

    def trim(self, n: int) -> int:
        return 0

    def make_mask(self, N, offset=None, return_array=True, window_size=None):
        if offset is None:
            offset = self.offset
        ws = window_size if window_size is not None else self.window_size
        # Causal + window-bounded mask. Use create_causal_mask directly to
        # avoid recursing through create_attention_mask -> cache.make_mask.
        return create_causal_mask(N, offset=offset, window_size=ws)

    def empty(self) -> bool:
        return self.offset == 0

    @classmethod
    def merge(cls, caches):
        """Convenience: merge a list of per-row V4Cache into one BatchV4Cache.

        Mirrors ``PoolingCache.merge`` (cache_extras.py:178) — the actual
        merge logic lives on BatchV4Cache so the two classes don't have
        circular imports.
        """
        return BatchV4Cache.merge(caches)


# ---------------------------------------------------------------------------
# BatchV4Cache — V4Cache + batch-aware metadata (BatchedEngine path)
# ---------------------------------------------------------------------------


def _concat_axis0(a, b, axis_seq):
    """Concatenate two state arrays on axis 0, padding the seq-dim to max.

    ``axis_seq`` is the axis index of the sequence dimension within the
    array (e.g. 2 for keys/values, 1 for compressed). If either input is
    None, the result is the other (no padding required since axis-0
    concatenation only happens between sibling BatchV4Caches and the
    state-machine arrays — which have no sequence axis — are concatenated
    via this helper too with ``axis_seq=None``).
    """
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    if axis_seq is None:
        # No seq dim to align — straight axis-0 concat.
        return mx.concatenate([a, b], axis=0)
    sa = a.shape[axis_seq]
    sb = b.shape[axis_seq]
    if sa == sb:
        return mx.concatenate([a, b], axis=0)
    if sa < sb:
        pad_shape = list(a.shape)
        pad_shape[axis_seq] = sb - sa
        pad = mx.zeros(tuple(pad_shape), dtype=a.dtype)
        a = mx.concatenate([a, pad], axis=axis_seq)
    else:
        pad_shape = list(b.shape)
        pad_shape[axis_seq] = sa - sb
        pad = mx.zeros(tuple(pad_shape), dtype=b.dtype)
        b = mx.concatenate([b, pad], axis=axis_seq)
    return mx.concatenate([a, b], axis=0)


# Per-field seq-axis map for the 8 V4Cache state arrays. Mirrors
# V4CacheHandler.get_state_axis_info — keep these two tables in lockstep.
_V4_CACHE_STATE_SEQ_AXES: tuple[Optional[int], ...] = (
    2,     # keys
    2,     # values
    1,     # compressed
    None,  # compressor_kv_state
    None,  # compressor_score_state
    1,     # indexer_compressed
    None,  # indexer_compressor_kv_state
    None,  # indexer_compressor_score_state
)


class BatchV4Cache(_BaseCache):
    """Batched per-layer V4 cache — V4Cache + left_padding awareness.

    Holds the same 8 arrays as V4Cache but with the batch axis preserved
    across concurrent requests. Per-row sliding-window slicing on decode
    is keyed by the running ``offset`` (single batch-wide scalar; all rows
    advance in lockstep through mlx-lm's continuous-batching scheduler,
    with ``left_padding`` arrays masking padded positions at the attention
    layer).

    Closest analogs: ``BatchPoolingCache`` (cache_extras.py:182) for the
    filter/extend/extract/merge surface, and ``V4Cache`` for the
    per-layer compressor field layout.
    """

    def __init__(self, layer_idx, compress_ratio, window_size, left_padding):
        super().__init__()
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio
        self.window_size = window_size
        # left_padding: list[int], one entry per row in the batch. Pure
        # Python list (not mx.array) so filter/extend can index-rewrite it
        # without round-tripping through MLX.
        self.left_padding = list(left_padding)
        batch_size = len(self.left_padding)
        # Same 8 array slots as V4Cache.
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.compressed: Optional[mx.array] = None
        self.compressor_kv_state: Optional[mx.array] = None
        self.compressor_score_state: Optional[mx.array] = None
        self.indexer_compressed: Optional[mx.array] = None
        self.indexer_compressor_kv_state: Optional[mx.array] = None
        self.indexer_compressor_score_state: Optional[mx.array] = None
        # Single batch-wide offset (rows advance in lockstep).
        self.offset = 0
        # Per-row scheduler bookkeeping. ``_lengths`` is the total length
        # each row will reach (set during prepare(); 2**31 sentinel means
        # "unknown / unbounded"). ``_processed`` is how many real
        # (non-padded) tokens each row has consumed so far.
        self._lengths = [2 ** 31] * batch_size
        self._processed = [0] * batch_size

    @classmethod
    def from_v4_cache(cls, c: "V4Cache", left_padding) -> "BatchV4Cache":
        """Build a BatchV4Cache mirroring an existing V4Cache.

        Used by ``generate_patch.to_batch_cache(c, left_padding)``. The
        source V4Cache may already hold prefilled state (keys/values
        populated, compressor fields None on layers without CSA). All 8
        fields are copied through; if ``c.keys`` is rank-3 it is expanded
        to rank-4 by tiling along a fresh batch axis sized to
        ``len(left_padding)``.
        """
        bc = cls(
            layer_idx=c.layer_idx,
            compress_ratio=c.compress_ratio,
            window_size=c.window_size,
            left_padding=list(left_padding),
        )
        B = len(bc.left_padding)
        # Carry the offset over so subsequent decode ticks know how far
        # the prefill got. (Single batch-wide offset — see __init__.)
        bc.offset = int(c.offset)

        def _adapt(arr):
            if arr is None:
                return None
            # If the source array already has a populated batch axis matching
            # ``B``, keep it as-is. If the leading axis is 1 (single stream)
            # tile to ``B``. Otherwise leave alone — the caller is responsible
            # for ensuring shape compatibility.
            if arr.shape[0] == B:
                return arr
            if arr.shape[0] == 1 and B != 1:
                reps = (B,) + (1,) * (arr.ndim - 1)
                return mx.tile(arr, reps)
            return arr

        for name in _V4_CACHE_STATE_ARRAY_SLOTS:
            setattr(bc, name, _adapt(getattr(c, name)))

        # Seed _processed from the source offset so the per-row scheduler
        # accounting starts aligned with the prefilled prefix length.
        bc._processed = [bc.offset] * B
        return bc

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append new keys/values; return the slice the attention call uses.

        Matches V4Cache.update_and_fetch's contract — prefill (S>1) returns
        the grown cache; decode (S==1) slices to the rightmost
        ``window_size`` tokens when offset exceeds the window. The batch
        axis is preserved on both inputs and outputs.
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = self.keys.shape[2]
        new_S = keys.shape[2]
        # Bump per-row processed counters in lockstep — continuous-batching
        # schedules all rows together; per-row right-padding is handled at
        # the attention layer via left_padding masks, not here.
        for i in range(len(self._processed)):
            self._processed[i] += new_S
        if new_S == 1 and self.offset > self.window_size:
            return (
                self.keys[..., -self.window_size:, :],
                self.values[..., -self.window_size:, :],
            )
        return self.keys, self.values

    @property
    def state(self):
        # Same 8-tuple as V4Cache — entries are None until first written.
        return tuple(getattr(self, name) for name in _V4_CACHE_STATE_ARRAY_SLOTS)

    @state.setter
    def state(self, value):
        # Match V4Cache.state setter tolerance: length-2 (legacy keys/values)
        # or length-8 (full state).
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"BatchV4Cache.state must be assigned a tuple, got "
                f"{type(value).__name__}"
            )
        if len(value) == 2:
            self.keys, self.values = value
        elif len(value) == 8:
            for name, v in zip(_V4_CACHE_STATE_ARRAY_SLOTS, value):
                setattr(self, name, v)
        else:
            raise ValueError(
                f"BatchV4Cache.state expects a 2- or 8-tuple, got length "
                f"{len(value)}"
            )

    @property
    def meta_state(self):
        # Tuple, not dict, for symmetry with BatchPoolingCache.meta_state.
        # Extra trailing fields beyond V4Cache's quadruple:
        # left_padding, _lengths, _processed (per-row scheduler state).
        return (
            self.layer_idx,
            self.compress_ratio,
            self.offset,
            self.window_size,
            list(self.left_padding),
            list(self._lengths),
            list(self._processed),
        )

    @meta_state.setter
    def meta_state(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"BatchV4Cache.meta_state must be assigned a tuple, got "
                f"{type(value).__name__}"
            )
        if len(value) == 4:
            # Legacy V4Cache scalars (ad-hoc reconstruct path). Per-row
            # arrays default to a single-row batch.
            self.layer_idx, self.compress_ratio, self.offset, self.window_size = value
            self.left_padding = [0]
            self._lengths = [2 ** 31]
            self._processed = [0]
        elif len(value) == 7:
            (
                self.layer_idx,
                self.compress_ratio,
                self.offset,
                self.window_size,
                left_padding,
                lengths,
                processed,
            ) = value
            self.left_padding = list(left_padding)
            self._lengths = list(lengths)
            self._processed = list(processed)
        else:
            raise ValueError(
                f"BatchV4Cache.meta_state expects a 4- or 7-tuple, got "
                f"length {len(value)}"
            )

    def is_trimmable(self) -> bool:
        # Same conservative stance as V4Cache (Phase 1).
        return False

    def trim(self, n: int) -> int:
        return 0

    def make_mask(self, N, offset=None, return_array=True, window_size=None):
        # Same shape as V4Cache.make_mask — causal + window. The mask is
        # broadcastable across the batch axis because create_causal_mask
        # produces a (N, offset+N) array independent of B. Per-row
        # right-padding is masked at the attention layer (via left_padding
        # arrays), not here.
        # TODO: For correctness on rows whose effective offset differs
        # from the batch-wide offset, the attention layer relies on
        # left_padding masks. If a future change moves that responsibility
        # into the cache, this method needs a per-row mask path.
        if offset is None:
            offset = self.offset
        ws = window_size if window_size is not None else self.window_size
        return create_causal_mask(N, offset=offset, window_size=ws)

    def empty(self) -> bool:
        return all(p == 0 for p in self._processed)

    @property
    def nbytes(self) -> int:
        total = 0
        for name in _V4_CACHE_STATE_ARRAY_SLOTS:
            arr = getattr(self, name)
            if arr is not None:
                total += arr.nbytes
        return total

    # ------------------------------------------------------------------
    # Batch operations: filter / extend / extract / merge.
    # Modeled directly on BatchPoolingCache (cache_extras.py:408-562).
    # ------------------------------------------------------------------

    def filter(self, batch_indices):
        """Subselect rows. ``batch_indices`` is either an mx.array of int
        row indices or a Python iterable of ints. All 8 state arrays are
        sliced on axis 0; the three Python lists are index-rewritten.
        """
        if isinstance(batch_indices, mx.array):
            idx_list = batch_indices.tolist()
        else:
            idx_list = list(batch_indices)

        for name in _V4_CACHE_STATE_ARRAY_SLOTS:
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, arr[batch_indices])

        self.left_padding = [self.left_padding[i] for i in idx_list]
        self._lengths = [self._lengths[i] for i in idx_list]
        self._processed = [self._processed[i] for i in idx_list]

    def extend(self, other: "BatchV4Cache") -> None:
        """Append rows from ``other`` to ``self``. Each state array is
        concatenated on axis 0; mismatched sequence dims are zero-padded
        to the max. Mirrors BatchPoolingCache.extend.
        """
        for name, seq_axis in zip(_V4_CACHE_STATE_ARRAY_SLOTS, _V4_CACHE_STATE_SEQ_AXES):
            a = getattr(self, name)
            b = getattr(other, name)
            setattr(self, name, _concat_axis0(a, b, seq_axis))

        self.left_padding = self.left_padding + list(other.left_padding)
        self._lengths = self._lengths + list(other._lengths)
        self._processed = self._processed + list(other._processed)
        # Offset stays single-batch-wide; if rows from `other` were at a
        # different stage, the scheduler is expected to align them before
        # this point (same assumption BatchPoolingCache.extend makes).
        self.offset = max(self.offset, int(other.offset))

    def extract(self, idx: int) -> "V4Cache":
        """Produce a single-row V4Cache for batch index ``idx``."""
        cache = V4Cache.from_state(
            layer_idx=self.layer_idx,
            compress_ratio=self.compress_ratio,
            offset=self.offset,
            window_size=self.window_size,
        )
        for name in _V4_CACHE_STATE_ARRAY_SLOTS:
            arr = getattr(self, name)
            if arr is not None:
                # Contiguous slice along axis 0 keeps shape rank intact.
                setattr(cache, name, mx.contiguous(arr[idx : idx + 1]))
        return cache

    @classmethod
    def merge(cls, caches) -> "BatchV4Cache":
        """Merge a list of single-row V4Cache instances into one
        BatchV4Cache. Mirrors BatchPoolingCache.merge.
        """
        if not caches:
            raise ValueError("BatchV4Cache.merge requires at least one cache")
        if not all(
            c.layer_idx == caches[0].layer_idx
            and c.compress_ratio == caches[0].compress_ratio
            and c.window_size == caches[0].window_size
            for c in caches
        ):
            raise ValueError(
                "BatchV4Cache.merge: all caches must share layer_idx, "
                "compress_ratio, and window_size"
            )
        B = len(caches)
        bc = cls(
            layer_idx=caches[0].layer_idx,
            compress_ratio=caches[0].compress_ratio,
            window_size=caches[0].window_size,
            left_padding=[0] * B,
        )
        # Carry the max offset across rows — single-batch-wide tracking,
        # with shorter rows masked at attention.
        bc.offset = max(int(c.offset) for c in caches)
        bc._processed = [int(c.offset) for c in caches]

        # For each of the 8 fields: find the max seq-dim across non-None
        # entries, allocate a zero-padded batch tensor, and write each row
        # into its slot. None on every row → leave field as None.
        for name, seq_axis in zip(_V4_CACHE_STATE_ARRAY_SLOTS, _V4_CACHE_STATE_SEQ_AXES):
            arrs = [getattr(c, name) for c in caches]
            if all(a is None for a in arrs):
                continue
            non_none = [a for a in arrs if a is not None]
            ref = non_none[0]
            if seq_axis is None:
                # State-machine field: shape is identical across rows or
                # we just stack them on axis 0 directly. Pad missing rows
                # with zeros sized like the reference.
                out_shape = (B,) + tuple(ref.shape[1:])
                out = mx.zeros(out_shape, dtype=ref.dtype)
                for i, a in enumerate(arrs):
                    if a is not None:
                        out[i] = a[0]
                setattr(bc, name, out)
                continue
            max_seq = max(a.shape[seq_axis] for a in non_none)
            # Build the output shape from the reference, overriding axis 0
            # (batch) and the seq-axis to the merged sizes.
            out_shape = list(ref.shape)
            out_shape[0] = B
            out_shape[seq_axis] = max_seq
            out = mx.zeros(tuple(out_shape), dtype=ref.dtype)
            for i, a in enumerate(arrs):
                if a is None:
                    continue
                # Slot the row's data into the corresponding axis-0
                # position; if the row is shorter on seq_axis, leave the
                # tail zero-padded (length-aware masks at attention handle
                # the padded positions).
                slc = [slice(None)] * out.ndim
                slc[0] = slice(i, i + 1)
                slc[seq_axis] = slice(0, a.shape[seq_axis])
                out[tuple(slc)] = a
            setattr(bc, name, out)
        return bc


# ---------------------------------------------------------------------------
# Cache rollback for speculative decoding (step 3 of MTP-native head wire-up)
# ---------------------------------------------------------------------------


class CacheCheckpoint:
    """Snapshot / restore for V4Cache state across a speculative step.

    V4Cache mutations all happen by replacing attribute references with new
    MLX arrays (V4Cache.update_and_fetch, _compressor_step's setattr writes).
    MLX arrays are immutable, so a snapshot only needs to save the existing
    references — no copy required. Restore writes the saved refs back.

    Usage:

        with CacheCheckpoint(caches) as ckpt:
            # run drafter for K steps + run validator
            if reject:
                ckpt.restore()      # roll back; old references re-bound
            # default exit: no-op (accept; speculative state persists)
    """

    # Re-use the shared array-slot tuple so any field added for serialization
    # automatically participates in spec-decode rollback (and vice versa).
    # ``offset`` is checkpointed too — V4Cache stores it on the instance and
    # mutates it inside update_and_fetch — but it's omitted from the array
    # slots since it flows through ``meta_state``.
    _SLOTS = ("offset",) + _V4_CACHE_STATE_ARRAY_SLOTS

    def __init__(self, caches: list[V4Cache]):
        self.caches = caches
        self.snaps: list[dict] = []

    def __enter__(self) -> "CacheCheckpoint":
        self.snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # no implicit rollback; caller decides via .restore()

    def snapshot(self) -> None:
        self.snaps = [
            {s: getattr(c, s) for s in self._SLOTS}
            for c in self.caches
        ]

    def restore(self) -> None:
        for c, snap in zip(self.caches, self.snaps):
            for s, v in snap.items():
                setattr(c, s, v)


# ---------------------------------------------------------------------------
# Sinkhorn helper for HC residual mixing
# ---------------------------------------------------------------------------


@mx.compile
def hc_split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
):
    """Port of kernel.hc_split_sinkhorn (lines 372-438 of inference/kernel.py).

    Args:
      mixes: (..., mix_hc) where mix_hc = (2 + hc_mult) * hc_mult.
      hc_scale: (3,)
      hc_base: (mix_hc,)

    Returns:
      pre:  (..., hc_mult)
      post: (..., hc_mult)
      comb: (..., hc_mult, hc_mult)  Sinkhorn-normalized
    """
    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre = mx.sigmoid(pre_logits) + eps

    post_logits = mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    post = 2.0 * mx.sigmoid(post_logits)

    comb_logits = mixes[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
    comb = mx.reshape(comb_logits, (*mixes.shape[:-1], hc_mult, hc_mult))

    # First normalize: row-softmax + eps, then col-normalize.
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    # Sinkhorn iterations: alternate row/col normalization.
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Walsh-Hadamard transform (radix-2 butterfly) along last axis.
# ---------------------------------------------------------------------------


@mx.compile
def walsh_hadamard_last(x: mx.array) -> mx.array:
    """Iterative radix-2 fast Walsh-Hadamard transform on the last axis.

    Returns x @ H_N (un-normalized; Sylvester construction). N must be a power
    of two. Caller should multiply by 1/sqrt(N) for the normalized variant.

    Shape-keyed mx.compile: at decode, only (batch=(1,), N=4096) and
    (batch=(1, top_k), N=2048) shapes are seen — two traces, both stable
    across decode steps. The Python-side log2(N) butterfly loop unrolls
    into a single fused graph per shape.
    """
    *batch, N = x.shape
    assert (N & (N - 1)) == 0, f"FWHT requires power-of-2 last dim, got {N}"
    h = 1
    y = x
    while h < N:
        # Reshape so blocks of 2*h consecutive entries are processed together.
        y = y.reshape(*batch, N // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = mx.stack([a + b, a - b], axis=-2)
        y = y.reshape(*batch, N)
        h *= 2
    return y


# ---------------------------------------------------------------------------
# JANGTQ routed-expert MoE kernel.
#
# Holds 2-bit MXTQ-packed routed-expert weights (uint32, 16 indices/elem)
# plus per-output-row L2 norms (bf16) of the Hadamard-rotated weight. At
# forward, applies randomized-Hadamard rotation to inputs (signs*x then
# Walsh-Hadamard / sqrt(N)), then for each top-k slot decodes the gathered
# active experts on the fly into bf16 and runs SwiGLU + down projection.
#
# Decode formula (validated against BF16 reference, cosine 0.94, sign-match 99.9%):
#   W_rotated[i, k] = tq_norms[i] * codebook_N[unpack_2bit(tq_packed)[i, k]]
#
# Rotation invariant (R = D_signs @ H / sqrt(N), R^T R = I):
#   x_rot @ W_rot^T = x @ W^T   so we matmul rotated-against-rotated.
# ---------------------------------------------------------------------------


def _unpack_2bit_last(packed: mx.array) -> mx.array:
    """uint32 (..., in/16) packed 2-bit indices, lo-bit-first slot order, →
    uint8 (..., in) with values in [0, 3]. Indices feed a (4,) codebook so
    uint8 is sufficient — saves 4x memory vs the prior int32 path, which
    matters at large (E, out, in) decode sizes.
    """
    in_packed = packed.shape[-1]
    in_dim = in_packed * 16
    shifts = mx.arange(16, dtype=mx.uint32) * mx.array(2, dtype=mx.uint32)
    mask = mx.array(0x3, dtype=mx.uint32)
    expanded = (packed[..., None] >> shifts) & mask
    return expanded.reshape(*packed.shape[:-1], in_dim).astype(mx.uint8)


def _decode_jangtq(packed: mx.array, norms: mx.array, codebook: mx.array) -> mx.array:
    """Decode batched 2-bit MXTQ routed-expert weights to bf16.

    packed:   (..., out, in/16) uint32
    norms:    (..., out) bf16 — per-row L2 of rotated weight
    codebook: (4,) fp32

    Returns (..., out, in) bf16.

    The codebook is cast to bf16 once and the multiply stays in bf16. This
    halves the intermediate footprint vs. promoting to fp32 — at decode
    sizes of (E, out, in) bf16 the fp32 detour was ~3x the dtype's bytes and
    doubled peak memory during the layer's decode + matmul pipeline.

    Kept as the validation reference for `_jangtq_fused_matmul` (synthetic-tiny
    cosine ≥ 0.999 gate). Not on the active forward path.
    """
    cb_bf16 = codebook if codebook.dtype == mx.bfloat16 else codebook.astype(mx.bfloat16)
    idx = _unpack_2bit_last(packed)                              # (..., out, in) int32
    return norms[..., None] * cb_bf16[idx]                       # (..., out, in) bf16


# ---------------------------------------------------------------------------
# Fused JANGTQ decode + gemv Metal kernel.
#
# Replaces the (decode → bf16 (N, top_k, O, K) transient → einsum) pair on
# the routed-expert hot path. Each threadgroup computes one out[n, k, o]:
# 32 lanes stride the K axis in 16-slot uint32 words, expand 2-bit codes in
# registers, fp32-accumulate, simd_sum-reduce, scale by per-row L2 norm,
# cast to T. The packed expert table is read with index inlining (no
# pre-gather), so the per-chunk transient is just (N, top_k, O) — the
# (N, top_k, O, K) bf16 weight materialization is gone.
#
# Hadamard rotation stays OUTSIDE this kernel. Inputs are pre-rotated x_rot
# / int_rot at the call site (see _batched_chunk). Re-applying signs or
# Walsh-Hadamard inside would silently mis-scale outputs and synthetic-tiny
# validation would not catch it (the reference makes the same outside-the-
# kernel assumption).
#
# Two call modes via shared_input template flag:
#   shared_input=1 — x is (N, K), broadcast across top_k slots (W1/W3).
#   shared_input=0 — x is (N, top_k, K), per-slot input (W2).
# ---------------------------------------------------------------------------


def _make_jangtq_fused_kernel():
    source = """
    constexpr uint LANES = 32;

    uint lane = thread_position_in_grid.x;       // 0..31 within the simdgroup
    uint flat = thread_position_in_grid.y;       // 0..(N*top_k*O_dim - 1)

    uint o = flat % O_dim;
    uint nk = flat / O_dim;
    uint k = nk % top_k;
    uint n = nk / top_k;

    uint eid = uint(indices[n * top_k + k]);
    uint xrow = shared_input ? n : (n * top_k + k);
    uint nwords = K_dim / 16;
    uint base_packed = (eid * O_dim + o) * nwords;
    uint base_x = xrow * K_dim;

    float acc = 0.0f;

    for (uint w_idx = lane; w_idx < nwords; w_idx += LANES) {
        uint w = packed[base_packed + w_idx];
        uint k_off = w_idx * 16u;
        for (uint j = 0u; j < 16u; j++) {
            uint c = (w >> (j * 2u)) & 0x3u;
            float xv = float(x[base_x + k_off + j]);
            float cv = float(codebook[c]);
            acc += xv * cv;
        }
    }

    float reduced = simd_sum(acc);

    if (lane == 0) {
        float nrm = float(norms[eid * O_dim + o]);
        out[(n * top_k + k) * O_dim + o] = T(reduced * nrm);
    }
    """
    return mx.fast.metal_kernel(
        name="jangtq_fused_decode_gemv",
        input_names=["x", "indices", "packed", "norms", "codebook"],
        output_names=["out"],
        source=source,
    )


_JANGTQ_FUSED_KERNEL = _make_jangtq_fused_kernel()


def _make_jangtq_fused_w1w3_kernel():
    source = """
    constexpr uint LANES = 32;

    uint lane = thread_position_in_grid.x;       // 0..31 within the simdgroup
    uint flat = thread_position_in_grid.y;       // 0..(N*O_dim - 1)

    uint o = flat % O_dim;
    uint n = flat / O_dim;
    uint nwords = K_dim / 16;
    uint base_x = n * K_dim;

    float acc1[8];
    float acc3[8];
    uint eids[8];
    uint bases[8];
    for (uint kk = 0u; kk < top_k; kk++) {
        uint eid = uint(indices[n * top_k + kk]);
        eids[kk] = eid;
        bases[kk] = (eid * O_dim + o) * nwords;
        acc1[kk] = 0.0f;
        acc3[kk] = 0.0f;
    }

    for (uint w_idx = lane; w_idx < nwords; w_idx += LANES) {
        uint ww1[8];
        uint ww3[8];
        for (uint kk = 0u; kk < top_k; kk++) {
            ww1[kk] = packed1[bases[kk] + w_idx];
            ww3[kk] = packed3[bases[kk] + w_idx];
        }
        uint k_off = w_idx * 16u;
        for (uint j = 0u; j < 16u; j++) {
            float xv = float(x[base_x + k_off + j]);
            uint sh = j * 2u;
            for (uint kk = 0u; kk < top_k; kk++) {
                uint c1 = (ww1[kk] >> sh) & 0x3u;
                uint c3 = (ww3[kk] >> sh) & 0x3u;
                acc1[kk] += xv * float(codebook[c1]);
                acc3[kk] += xv * float(codebook[c3]);
            }
        }
    }

    for (uint kk = 0u; kk < top_k; kk++) {
        float red1 = simd_sum(acc1[kk]);
        float red3 = simd_sum(acc3[kk]);
        if (lane == 0) {
            uint eid = eids[kk];
            uint out_idx = (n * top_k + kk) * O_dim + o;
            gate[out_idx] = T(red1 * float(norms1[eid * O_dim + o]));
            up[out_idx] = T(red3 * float(norms3[eid * O_dim + o]));
        }
    }
    """
    return mx.fast.metal_kernel(
        name="jangtq_fused_w1w3_decode_gemv",
        input_names=[
            "x", "indices", "packed1", "norms1", "packed3", "norms3", "codebook",
        ],
        output_names=["gate", "up"],
        source=source,
    )


_JANGTQ_FUSED_W1W3_KERNEL = _make_jangtq_fused_w1w3_kernel()


def _jangtq_fused_matmul(
    x: mx.array,
    indices: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
    shared_input: bool,
) -> mx.array:
    """Fused JANGTQ decode + gemv. See module-level kernel docstring above.

    shared_input=True:  x shape (N, K).         Output (N, top_k, O).
    shared_input=False: x shape (N, top_k, K).  Output (N, top_k, O).

    The two modes exist because the routed-expert MoE forward (see
    `_batched_chunk`) feeds W1/W3 and W2 from structurally different tensors.
    W1 (gate) and W3 (up) both consume `x_rot` of shape (N, H): routing only
    fans out at the weight-gather step, so the same rotated input is reused
    across every top_k slot (shared_input=True). W2 (down) consumes `int_rot`,
    the post-SwiGLU `silu(gate) * up`, which is already per-slot of shape
    (N, top_k, I) — there is nothing to broadcast (shared_input=False). Do
    not collapse the modes; broadcasting int_rot would silently mix slots.

    indices: (N, top_k) integer expert ids; cast to uint32 at the boundary.
    packed:  (n_experts, O, K/16) uint32.
    norms:   (n_experts, O)       same dtype as x (bf16 in production).
    codebook: (4,) — cast to x.dtype if needed.
    """
    N, top_k = indices.shape[0], indices.shape[1]
    n_experts, O_dim = norms.shape
    K_dim = packed.shape[-1] * 16
    cb = codebook if codebook.dtype == x.dtype else codebook.astype(x.dtype)
    idx32 = indices.astype(mx.uint32)

    return _JANGTQ_FUSED_KERNEL(
        inputs=[x, idx32, packed, norms, cb],
        template=[
            ("T", x.dtype),
            ("K_dim", K_dim),
            ("O_dim", O_dim),
            ("top_k", top_k),
            ("shared_input", 1 if shared_input else 0),
        ],
        grid=(32, N * top_k * O_dim, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N, top_k, O_dim)],
        output_dtypes=[x.dtype],
    )[0]


def _jangtq_fused_w1w3_matmul(
    x: mx.array,
    indices: mx.array,
    packed1: mx.array,
    norms1: mx.array,
    packed3: mx.array,
    norms3: mx.array,
    codebook: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fused W1(gate)+W3(up) JANGTQ decode+gemv for shared-input MoE slots.

    W1 and W3 consume the same rotated input and routed expert indices. The
    kernel computes all top-k slots for each `(token, output_row)` so each x
    element is read once and reused across the active experts, while decoding
    both packed weight streams. This preserves the exact JANGTQ codebook math.
    """
    N, top_k = indices.shape[0], indices.shape[1]
    _n_experts, O_dim = norms1.shape
    K_dim = packed1.shape[-1] * 16
    if top_k > 8:
        raise ValueError("_jangtq_fused_w1w3_matmul supports top_k <= 8")
    cb = codebook if codebook.dtype == x.dtype else codebook.astype(x.dtype)
    idx32 = indices.astype(mx.uint32)

    gate, up = _JANGTQ_FUSED_W1W3_KERNEL(
        inputs=[x, idx32, packed1, norms1, packed3, norms3, cb],
        template=[
            ("T", x.dtype),
            ("K_dim", K_dim),
            ("O_dim", O_dim),
            ("top_k", top_k),
        ],
        grid=(32, N * O_dim, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N, top_k, O_dim), (N, top_k, O_dim)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return gate, up


@mx.compile
def _jangtq_batched_chunk_compiled(
    x_flat: mx.array,
    indices: mx.array,
    weights: mx.array,
    w1_packed: mx.array,
    w1_norms: mx.array,
    w3_packed: mx.array,
    w3_norms: mx.array,
    w2_packed: mx.array,
    w2_norms: mx.array,
    cb_4096: mx.array,
    cb_2048: mx.array,
    signs_4096: mx.array,
    signs_2048: mx.array,
    inv_sqrt_H: mx.array,
    inv_sqrt_I: mx.array,
    swiglu_limit: float,
) -> mx.array:
    """Compiled body of one MoE chunk forward — fuses Python-side dispatch
    between FWHT, the 3 fused-decode+gemv Metal kernels, SwiGLU, and the
    weighted slot reduction. Shape-keyed: at decode N=1 fixed.
    """
    x_rot = walsh_hadamard_last(x_flat * signs_4096) * inv_sqrt_H        # (N, H)
    gate, up = _jangtq_fused_w1w3_matmul(
        x_rot, indices, w1_packed, w1_norms, w3_packed, w3_norms, cb_4096,
    )                                                                     # (N, top_k, I)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    intermediate = nn.silu(gate) * up                                     # (N, top_k, I)
    int_rot = walsh_hadamard_last(intermediate * signs_2048) * inv_sqrt_I
    contrib = _jangtq_fused_matmul(
        int_rot, indices, w2_packed, w2_norms, cb_2048, shared_input=False,
    )                                                                     # (N, top_k, H)
    return (weights[..., None].astype(x_flat.dtype) * contrib).sum(axis=1)


class JANGTQRoutedExperts(nn.Module):
    """Routed-expert MoE path for the JANGTQ 2-bit MXTQ format.

    Stacks per-layer routed-expert weights along the expert axis:
      w{1,3}_tq_packed: (n_experts, moe_inter, hidden//16) uint32   # gate / up
      w{1,3}_tq_norms:  (n_experts, moe_inter)            bf16
      w2_tq_packed:     (n_experts, hidden, moe_inter//16) uint32   # down
      w2_tq_norms:      (n_experts, hidden)               bf16

    Forward (called by DeepseekV4MoE):
      x_flat:  (N, hidden)
      indices: (N, top_k) integer expert ids
      weights: (N, top_k) router weights (post sqrtsoftplus + scaling)
      jangtq:  dict with cb_4096, signs_4096, cb_2048, signs_2048

    Returns (N, hidden) routed contribution (weighted sum across slots).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts = args.n_routed_experts
        self.hidden = args.hidden_size                # 4096
        self.inter = args.moe_intermediate_size       # 2048
        self.swiglu_limit = args.swiglu_limit
        H, I = self.hidden, self.inter
        self.w1_tq_packed = mx.zeros((self.n_experts, I, H // 16), dtype=mx.uint32)
        self.w1_tq_norms = mx.zeros((self.n_experts, I), dtype=mx.bfloat16)
        self.w3_tq_packed = mx.zeros((self.n_experts, I, H // 16), dtype=mx.uint32)
        self.w3_tq_norms = mx.zeros((self.n_experts, I), dtype=mx.bfloat16)
        self.w2_tq_packed = mx.zeros((self.n_experts, H, I // 16), dtype=mx.uint32)
        self.w2_tq_norms = mx.zeros((self.n_experts, H), dtype=mx.bfloat16)

    # Per-chunk token batch for the dispatch path. The active path uses fused
    # decode+gemv kernels, so larger chunks mostly reduce Python/Metal launch
    # count rather than materializing giant decoded-weight tensors. 2026-05-08
    # P=1024 sweep: CHUNK_N=32 beat 8/128/256 (123.55 tok/s vs 119.74
    # baseline), while 128/256 regressed slightly from 32.
    CHUNK_N = 32

    def _batched_chunk(
        self,
        x_flat: mx.array,           # (N_chunk, H)
        indices: mx.array,          # (N_chunk, top_k)
        weights: mx.array,          # (N_chunk, top_k)
        cb_4096: mx.array,
        cb_2048: mx.array,
        signs_4096: mx.array,
        signs_2048: mx.array,
        inv_sqrt_H: mx.array,
        inv_sqrt_I: mx.array,
    ) -> mx.array:
        return _jangtq_batched_chunk_compiled(
            x_flat, indices, weights,
            self.w1_tq_packed, self.w1_tq_norms,
            self.w3_tq_packed, self.w3_tq_norms,
            self.w2_tq_packed, self.w2_tq_norms,
            cb_4096, cb_2048, signs_4096, signs_2048,
            inv_sqrt_H, inv_sqrt_I,
            self.swiglu_limit,
        )

    def __call__(
        self,
        x_flat: mx.array,           # (N, hidden)
        indices: mx.array,          # (N, top_k) integer
        weights: mx.array,          # (N, top_k) bf16
        jangtq: dict[str, mx.array],
    ) -> mx.array:
        N, H = x_flat.shape
        cb_4096 = jangtq["cb_4096"]
        cb_2048 = jangtq["cb_2048"]
        signs_4096 = jangtq["signs_4096"].astype(x_flat.dtype)
        signs_2048 = jangtq["signs_2048"].astype(x_flat.dtype)
        inv_sqrt_H = mx.array(1.0 / (H ** 0.5), dtype=x_flat.dtype)
        inv_sqrt_I = mx.array(1.0 / (self.inter ** 0.5), dtype=x_flat.dtype)

        if N <= self.CHUNK_N:
            return self._batched_chunk(
                x_flat, indices, weights,
                cb_4096, cb_2048, signs_4096, signs_2048, inv_sqrt_H, inv_sqrt_I,
            )

        chunks: list[mx.array] = []
        for start in range(0, N, self.CHUNK_N):
            end = min(start + self.CHUNK_N, N)
            chunks.append(self._batched_chunk(
                x_flat[start:end], indices[start:end], weights[start:end],
                cb_4096, cb_2048, signs_4096, signs_2048, inv_sqrt_H, inv_sqrt_I,
            ))
        return mx.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# V4 long-range-attention helpers
#
# Ports the small support functions from `inference/model.py` that the
# Compressor / Indexer / sparse-attn path consume:
#   - apply_rotary_emb (model.py:232-244): real-valued rotation.
#     Reference uses torch.view_as_complex; MLX has no native complex, so we
#     represent freqs_cis as (..., rd//2, 2) holding (cos, sin) and do the
#     complex-multiply by hand.
#   - precompute_freqs_cis (subset of model.py:194-229): simple base-rope
#     without YaRN, sufficient for the compressed-stream's compress_rope_theta.
#   - get_window_topk_idxs (model.py:255-265): per-query indices into the
#     window slot of the KV passed to sparse_attn (or into the freshly
#     projected kv during prefill).
#   - get_compress_topk_idxs (model.py:269-276): identity-like fallback
#     compressed-stream indices used when self.indexer is None
#     (compress_ratio==128 HCA layers).
#   - sparse_attn (model.py:355-368, kernel.py:276-368): pure-MLX prototype
#     of the TileLang sparse attention kernel. Phase 1 — semantically
#     equivalent to the kernel, materializes everything (no flash-style
#     online softmax). The Metal port is a Phase 2 perf concern.
# ---------------------------------------------------------------------------


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array, inverse: bool = False) -> mx.array:
    """Real-valued port of `apply_rotary_emb` (model.py:232-244).

    Args:
      x: (..., S, rd) where rd is even. Last dim holds (re, im) interleaved
         pairs of the same complex view the reference takes.
      freqs_cis: (S, rd//2, 2) where the last dim is (cos, sin).
      inverse: if True, apply the conjugate rotation (de-rotation).
    Returns:
      x rotated, same shape and dtype as input.
    """
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    *lead, S, rd = x.shape
    x = x.reshape(*lead, S, rd // 2, 2)
    x_re = x[..., 0]
    x_im = x[..., 1]
    f_re = freqs_cis[..., 0]
    f_im = freqs_cis[..., 1]
    if inverse:
        f_im = -f_im
    # Broadcast freqs to align (S, rd//2) against the last two axes of x.
    if x_re.ndim == 3:
        # (B, S, rd//2): freqs -> (1, S, rd//2)
        f_re = mx.expand_dims(f_re, axis=0)
        f_im = mx.expand_dims(f_im, axis=0)
    elif x_re.ndim == 4:
        # (B, S, H, rd//2): freqs -> (1, S, 1, rd//2)
        f_re = mx.expand_dims(mx.expand_dims(f_re, axis=0), axis=2)
        f_im = mx.expand_dims(mx.expand_dims(f_im, axis=0), axis=2)
    elif x_re.ndim != 2:
        raise ValueError(f"apply_rotary_emb: unsupported x ndim {x_re.ndim}")
    out_re = x_re * f_re - x_im * f_im
    out_im = x_re * f_im + x_im * f_re
    out = mx.stack([out_re, out_im], axis=-1).reshape(*lead, S, rd)
    return out.astype(orig_dtype)


def precompute_freqs_cis(rope_head_dim: int, max_seq_len: int, theta: float) -> mx.array:
    """Simple base-rope freqs (no YaRN). Returns (max_seq_len, rope_head_dim//2, 2)
    where the last dim holds (cos, sin). Phase 1 uses this for compressed-stream
    rope; the regular per-token rope still goes through self.rope (mlx_lm)."""
    half = rope_head_dim // 2
    inv_freqs = 1.0 / (theta ** (mx.arange(0, rope_head_dim, 2, dtype=mx.float32) / rope_head_dim))
    t = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freqs)                                   # (max_seq_len, half)
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    return mx.stack([cos, sin], axis=-1)                              # (max_seq_len, half, 2)


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int) -> mx.array:
    """Port of model.py:255-265. Returns (bsz, m, window_size) int32.
    During prefill (start_pos==0), m == seqlen and the matrix is upper-
    triangular: row i holds [max(0, i+1-window), ..., i] right-padded with
    -1. During decode (start_pos>0), m == 1 and the row indexes the window
    slot of the KV passed to sparse_attn.
    """
    win = window_size
    if start_pos >= win - 1:
        sp = start_pos % win
        # Circular: [sp+1, sp+2, ..., win-1, 0, 1, ..., sp]
        first = mx.arange(sp + 1, win, dtype=mx.int32)
        second = mx.arange(0, sp + 1, dtype=mx.int32)
        matrix = mx.concatenate([first, second], axis=0)               # (win,)
        matrix = matrix.reshape(1, win)                                # (m=1, win)
    elif start_pos > 0:
        # Pad right with -1 to length win.
        head = mx.arange(start_pos + 1, dtype=mx.int32)
        pad = mx.full((win - start_pos - 1,), vals=-1, dtype=mx.int32)
        matrix = mx.concatenate([head, pad], axis=0).reshape(1, win)
    else:
        # Prefill: per-query window indices.
        base = mx.arange(seqlen, dtype=mx.int32).reshape(seqlen, 1)
        clamped = mx.maximum(base - win + 1, mx.array(0, dtype=mx.int32))
        offsets = mx.arange(min(seqlen, win), dtype=mx.int32).reshape(1, -1)
        matrix = clamped + offsets                                     # (seqlen, min(seqlen, win))
        # If seqlen < win, right-pad with -1 to width win.
        if seqlen < win:
            pad = mx.full((seqlen, win - seqlen), vals=-1, dtype=mx.int32)
            matrix = mx.concatenate([matrix, pad], axis=1)
        # Mask future positions (matrix > base) to -1.
        matrix = mx.where(matrix > base, mx.array(-1, dtype=mx.int32), matrix)
    # Broadcast to (bsz, m, win).
    return mx.broadcast_to(matrix[None, ...], (bsz, matrix.shape[0], win))


def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int) -> mx.array:
    """Port of model.py:269-276. Identity-like fallback used when
    self.indexer is None (compress_ratio==128 HCA layers).

    Three regimes:
      - Decode (start_pos > 0, seqlen == 1): single query at absolute
        position start_pos; returns (B, 1, n) where n = (start_pos+1)//ratio.
      - Prefill (start_pos == 0): S queries at positions [0, S); returns
        (B, S, n) with per-row causal masking via -1 sentinels.
      - Verify (start_pos > 0, seqlen > 1): chunked-prefill continuation —
        S queries at positions [start_pos, start_pos+S); returns
        (B, S, n) where n = (start_pos+S)//ratio with the same per-row
        causal masking (row i sees indices [0, (start_pos+i+1)//ratio)).
    """
    if start_pos > 0 and seqlen == 1:
        n = (start_pos + 1) // ratio
        matrix = mx.arange(n, dtype=mx.int32) + offset                 # (n,)
        matrix = matrix.reshape(1, n)                                  # (m=1, n)
        return mx.broadcast_to(matrix[None, ...], (bsz, 1, n))
    # Prefill (start_pos==0) or verify (start_pos>0, seqlen>1). Both use
    # the same per-row causal pattern; start_pos==0 collapses to the
    # original prefill formula.
    n = (start_pos + seqlen) // ratio
    if n == 0:
        return mx.zeros((bsz, seqlen, 0), dtype=mx.int32)
    base = mx.broadcast_to(mx.arange(n, dtype=mx.int32).reshape(1, n), (seqlen, n))
    threshold = (
        mx.arange(start_pos + 1, start_pos + seqlen + 1, dtype=mx.int32) // ratio
    ).reshape(seqlen, 1)
    masked = base >= threshold
    matrix = mx.where(masked, mx.array(-1, dtype=mx.int32), base + offset)   # (seqlen, n)
    return mx.broadcast_to(matrix[None, ...], (bsz, seqlen, n))


def sparse_attn(
    q: mx.array,          # (B, S, h, d)  bf16
    kv: mx.array,         # (B, N, d)     bf16  (single shared KV head)
    attn_sink: mx.array,  # (h,)          fp32  per-head learnable sink
    topk_idxs: mx.array,  # (B, S, K)     int32  (-1 = padded/invalid)
    scale: float,
) -> mx.array:
    """Pure-MLX sparse attention with per-head sink (Phase 1 prototype).

    Mirrors `kernel.py:276-368` (TileLang) semantics: gather KV at top-k
    indices, fp32 stable softmax with sink contributing only to the
    denominator (kernel.py:345-348), output bf16.
    """
    B, S, h, d = q.shape
    _, N, d_kv = kv.shape
    assert d == d_kv, f"q.d={d} vs kv.d={d_kv}"
    K = topk_idxs.shape[-1]

    # Gather kv at top-k positions, zeroing -1 lanes.
    valid = topk_idxs >= 0                                             # (B, S, K) bool
    safe_idx = mx.maximum(topk_idxs, 0)                                # (B, S, K) int32
    kv_expanded = mx.broadcast_to(kv[:, None, :, :], (B, S, N, d))     # (B, S, N, d)
    gather_idx = mx.broadcast_to(safe_idx[..., None], (B, S, K, d))    # (B, S, K, d)
    kv_gathered = mx.take_along_axis(kv_expanded, gather_idx, axis=2)  # (B, S, K, d)
    kv_gathered = mx.where(valid[..., None], kv_gathered, mx.zeros_like(kv_gathered))

    # Scores in fp32 (matches kernel's FP32 acc_s fragment).
    q_f32 = q.astype(mx.float32)
    kv_g_f32 = kv_gathered.astype(mx.float32)
    scores = mx.einsum("bshd,bskd->bshk", q_f32, kv_g_f32) * scale     # (B, S, h, K)
    neg_inf = mx.array(-mx.inf, dtype=mx.float32)
    scores = mx.where(valid[:, :, None, :], scores, neg_inf)

    # Stable softmax with sink-in-denominator-only.
    m = mx.max(scores, axis=-1, keepdims=True)                          # (B, S, h, 1)
    exps = mx.exp(scores - m)                                           # (B, S, h, K)
    sum_exps = mx.sum(exps, axis=-1, keepdims=True)                     # (B, S, h, 1)
    sink = attn_sink.astype(mx.float32).reshape(1, 1, h, 1)
    denom = sum_exps + mx.exp(sink - m)                                 # (B, S, h, 1)
    weights = exps / denom                                              # (B, S, h, K)

    out_f32 = mx.einsum("bshk,bskd->bshd", weights, kv_g_f32)
    return out_f32.astype(q.dtype)


# ---------------------------------------------------------------------------
# CSA Compressor / Indexer
# ---------------------------------------------------------------------------


def _compressed_freqs_cis(positions: mx.array, rope_head_dim: int, theta: float) -> mx.array:
    """Compute (cos, sin) freqs at given positions for compressed-stream rope.
    Returns (P, rope_head_dim//2, 2). Phase 1 — no YaRN. Computed on the fly
    each compressor.forward to avoid holding a 1M-position lookup in memory."""
    inv_freqs = 1.0 / (theta ** (mx.arange(0, rope_head_dim, 2, dtype=mx.float32) / rope_head_dim))
    angles = mx.outer(positions.astype(mx.float32), inv_freqs)         # (P, rd//2)
    return mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)         # (P, rd//2, 2)


def _splice(t: mx.array, lo: int, hi: int, replacement: mx.array) -> mx.array:
    """Functional splice: t[lo:hi] = replacement (along axis=1).
    Returns a new tensor; original is untouched. lo/hi are along axis 1.
    """
    return mx.concatenate([t[:, :lo], replacement, t[:, hi:]], axis=1)


def _compressor_step(
    comp,                            # has .compress_ratio, .overlap, .head_dim, .rope_head_dim,
                                     #     .compress_rope_theta, .max_compressed_positions,
                                     #     .ape, .wkv, .wgate, .norm
    x: mx.array,                     # (B, S, hidden)
    start_pos: int,
    cache,                           # V4Cache
    slot_compressed: str,            # cache attribute name for compressed kv buffer
    slot_kv_state: str,              # cache attribute name for kv state buffer
    slot_score_state: str,           # cache attribute name for score state buffer
) -> Optional[mx.array]:
    """Phase 1 port of inference/model.py::Compressor.forward (lines 316-377).

    Skipped vs reference (Phase 1 simplifications):
      - rotate_activation (model.py:368-369): Walsh-Hadamard rotation, paired
        with Indexer's q-side rotation that cancels in the einsum. Skipping
        both is equivalent for the einsum.
      - fp4_act_quant / act_quant (model.py:370, :372): QAT-noise; cache
        stores bf16 either way per the inline reference comment at :527.
    """
    bsz, seqlen, _ = x.shape
    ratio = comp.compress_ratio
    overlap = comp.overlap
    d = comp.head_dim
    rd = comp.rope_head_dim
    coff = 2 if overlap else 1
    in_dtype = x.dtype

    # Lazy-allocate state and compressed-cache buffers on first call.
    kv_state = getattr(cache, slot_kv_state)
    score_state = getattr(cache, slot_score_state)
    if kv_state is None:
        kv_state = mx.zeros((bsz, coff * ratio, coff * d), dtype=mx.float32)
        score_state = mx.full((bsz, coff * ratio, coff * d), vals=-mx.inf, dtype=mx.float32)
    compressed_buf = getattr(cache, slot_compressed)
    if compressed_buf is None:
        compressed_buf = mx.zeros(
            (bsz, comp.max_compressed_positions, d), dtype=mx.bfloat16
        )

    # Linears + ape addition all in fp32 (reference: model.py:322 "compression need fp32").
    x_f32 = x.astype(mx.float32)
    kv = comp.wkv(x_f32)
    score = comp.wgate(x_f32)
    ape_f32 = comp.ape.astype(mx.float32)

    if start_pos > 0 and seqlen > 1:
        # ---------- verify regime (third regime) ----------
        # Unroll the decode arm across `seqlen` tokens. By construction this
        # produces the same sequence of fp32 reductions as `seqlen` sequential
        # S=1 decode calls — i.e. bit-equal to the reference path. Each
        # iteration advances `start_pos` by 1 and uses the i-th column slice
        # of the precomputed (B, S, *) `kv` and `score` tensors.
        #
        # DO NOT route this through the prefill arm: that arm clobbers
        # kv_state/score_state (`_splice` from slot 0..ratio) instead of
        # merging with prior content; only the decode arm carries the
        # gated-pool state machine across calls.
        emissions: list = []  # list of (cslot:int, emit_pre_norm: mx.array)
        for i in range(seqlen):
            sp_i = start_pos + i
            kv_i = kv[:, i : i + 1]                                            # (B, 1, 2*d)
            score_i = score[:, i : i + 1] + ape_f32[sp_i % ratio]              # (B, 1, 2*d)
            should_compress_i = (sp_i + 1) % ratio == 0
            if overlap:
                slot = ratio + sp_i % ratio
                kv_state = _splice(
                    kv_state, slot, slot + 1, kv_i.astype(kv_state.dtype),
                )
                score_state = _splice(
                    score_state, slot, slot + 1, score_i.astype(score_state.dtype),
                )
                if should_compress_i:
                    kv_state_a = kv_state[:, :ratio, :d]
                    kv_state_b = kv_state[:, ratio:, d:]
                    sc_state_a = score_state[:, :ratio, :d]
                    sc_state_b = score_state[:, ratio:, d:]
                    kv_combined = mx.concatenate([kv_state_a, kv_state_b], axis=1)
                    sc_combined = mx.concatenate([sc_state_a, sc_state_b], axis=1)
                    emit = (kv_combined * mx.softmax(sc_combined, axis=1)).sum(
                        axis=1, keepdims=True,
                    )
                    emissions.append((sp_i // ratio, emit))
                    # Roll: kv_state[:ratio] = kv_state[ratio:]; second half preserved.
                    rolled_kv = mx.concatenate(
                        [kv_state[:, ratio:], kv_state[:, ratio:]], axis=1,
                    )
                    rolled_sc = mx.concatenate(
                        [score_state[:, ratio:], score_state[:, ratio:]], axis=1,
                    )
                    kv_state = rolled_kv
                    score_state = rolled_sc
            else:
                slot = sp_i % ratio
                kv_state = _splice(
                    kv_state, slot, slot + 1, kv_i.astype(kv_state.dtype),
                )
                score_state = _splice(
                    score_state, slot, slot + 1, score_i.astype(score_state.dtype),
                )
                if should_compress_i:
                    emit = (kv_state * mx.softmax(score_state, axis=1)).sum(
                        axis=1, keepdims=True,
                    )
                    emissions.append((sp_i // ratio, emit))

        # State writeback (regardless of emissions). Functional setattr to
        # preserve the V4Cache replacement semantics (CacheCheckpoint relies
        # on this).
        setattr(cache, slot_kv_state, kv_state)
        setattr(cache, slot_score_state, score_state)

        if not emissions:
            return None

        # Norm + compressed-stream rope per emission, then write to the
        # compressed buffer at the per-emission cslot. Position used for the
        # rope is `cslot * ratio` (= start_pos + i + 1 - ratio when
        # should_compress_i fires for the i-th token), matching the existing
        # decode arm's `start_pos + 1 - ratio`.
        out_chunks = []
        for cslot, emit in emissions:
            emit_n = comp.norm(emit.astype(in_dtype))                          # (B, 1, d)
            position = mx.array([cslot * ratio], dtype=mx.float32)             # (1,)
            freqs_cis = _compressed_freqs_cis(position, rd, comp.compress_rope_theta)
            rope_tail = apply_rotary_emb(emit_n[..., -rd:], freqs_cis)
            emit_final = mx.concatenate(
                [emit_n[..., :-rd], rope_tail.astype(emit_n.dtype)], axis=-1,
            )
            compressed_buf = _splice(
                compressed_buf, cslot, cslot + 1,
                emit_final.astype(compressed_buf.dtype),
            )
            out_chunks.append(emit_final)
        setattr(cache, slot_compressed, compressed_buf)
        return mx.concatenate(out_chunks, axis=1)                              # (B, M, d)

    if start_pos == 0:
        # ---------- prefill path (model.py:325-342) ----------
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset_state = ratio if overlap else 0

        # 330-332: seed overlap-state's first-half from the trailing `ratio` of prefill.
        if overlap and cutoff >= ratio:
            kv_state = _splice(
                kv_state, 0, ratio,
                kv[:, cutoff - ratio : cutoff].astype(kv_state.dtype),
            )
            score_state = _splice(
                score_state, 0, ratio,
                (score[:, cutoff - ratio : cutoff] + ape_f32).astype(score_state.dtype),
            )

        # 333-336: stash trailing remainder (post-cutoff) into state, trim kv/score.
        if remainder > 0:
            kv_state = _splice(
                kv_state, offset_state, offset_state + remainder,
                kv[:, cutoff : cutoff + remainder].astype(kv_state.dtype),
            )
            score_state = _splice(
                score_state, offset_state, offset_state + remainder,
                (score[:, cutoff:] + ape_f32[:remainder]).astype(score_state.dtype),
            )
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]

        # 337-342: reshape into windows, add APE, optional overlap remap, gated pool.
        n_windows = cutoff // ratio if cutoff > 0 else 0
        kv_w = kv.reshape(bsz, n_windows, ratio, kv.shape[-1])
        score_w = score.reshape(bsz, n_windows, ratio, score.shape[-1]) + ape_f32
        if overlap:
            # overlap_transform: build (B, n_windows, 2*ratio, d) from (B, n_windows, ratio, 2*d).
            #   right half: tensor[..., d:]      -> (B, n_windows, ratio, d)
            #   left half:  tensor[:, :-1, :, :d] padded with `value` at row 0
            right_kv = kv_w[..., d:]
            right_sc = score_w[..., d:]
            pad_kv = mx.zeros((bsz, 1, ratio, d), dtype=kv_w.dtype)
            pad_sc = mx.full((bsz, 1, ratio, d), vals=float("-inf"), dtype=score_w.dtype)
            left_kv = mx.concatenate([pad_kv, kv_w[:, :-1, :, :d]], axis=1)
            left_sc = mx.concatenate([pad_sc, score_w[:, :-1, :, :d]], axis=1)
            kv_w = mx.concatenate([left_kv, right_kv], axis=2)                 # (B, n, 2r, d)
            score_w = mx.concatenate([left_sc, right_sc], axis=2)              # (B, n, 2r, d)
        kv = (kv_w * mx.softmax(score_w, axis=2)).sum(axis=2)                  # (B, n_windows, d)

    else:
        # ---------- decode path (model.py:343-359) ----------
        should_compress = (start_pos + 1) % ratio == 0
        score = score + ape_f32[start_pos % ratio]
        if overlap:
            slot = ratio + start_pos % ratio
            kv_state = _splice(
                kv_state, slot, slot + 1,
                mx.expand_dims(kv[:, 0], axis=1).astype(kv_state.dtype),
            )
            score_state = _splice(
                score_state, slot, slot + 1,
                mx.expand_dims(score[:, 0], axis=1).astype(score_state.dtype),
            )
            if should_compress:
                # Concat first-half-`:d` and second-half-`d:` into (B, 2*ratio, d).
                kv_state_a = kv_state[:, :ratio, :d]
                kv_state_b = kv_state[:, ratio:, d:]
                sc_state_a = score_state[:, :ratio, :d]
                sc_state_b = score_state[:, ratio:, d:]
                kv_combined = mx.concatenate([kv_state_a, kv_state_b], axis=1)
                sc_combined = mx.concatenate([sc_state_a, sc_state_b], axis=1)
                kv = (kv_combined * mx.softmax(sc_combined, axis=1)).sum(axis=1, keepdims=True)
                # Roll: kv_state[:ratio] = kv_state[ratio:]; second half preserved.
                rolled_kv = mx.concatenate([kv_state[:, ratio:], kv_state[:, ratio:]], axis=1)
                rolled_sc = mx.concatenate([score_state[:, ratio:], score_state[:, ratio:]], axis=1)
                kv_state = rolled_kv
                score_state = rolled_sc
        else:
            slot = start_pos % ratio
            kv_state = _splice(
                kv_state, slot, slot + 1,
                mx.expand_dims(kv[:, 0], axis=1).astype(kv_state.dtype),
            )
            score_state = _splice(
                score_state, slot, slot + 1,
                mx.expand_dims(score[:, 0], axis=1).astype(score_state.dtype),
            )
            if should_compress:
                kv = (kv_state * mx.softmax(score_state, axis=1)).sum(axis=1, keepdims=True)

    # Write state back to cache (regardless of should_compress).
    setattr(cache, slot_kv_state, kv_state)
    setattr(cache, slot_score_state, score_state)

    if not should_compress:
        return None

    # Norm + compressed-stream rope on the rope-tail.
    kv = comp.norm(kv.astype(in_dtype))
    if start_pos == 0:
        positions = mx.arange(0, cutoff, ratio, dtype=mx.float32)              # (n_windows,)
    else:
        positions = mx.array([start_pos + 1 - ratio], dtype=mx.float32)        # (1,)
    freqs_cis = _compressed_freqs_cis(positions, rd, comp.compress_rope_theta)
    rope_tail = apply_rotary_emb(kv[..., -rd:], freqs_cis)
    kv = mx.concatenate([kv[..., :-rd], rope_tail.astype(kv.dtype)], axis=-1)

    # Phase 1: SKIP rotate_activation, fp4_act_quant, act_quant.

    # Cache write into the compressed buffer.
    if start_pos == 0:
        n_compressed = seqlen // ratio
        compressed_buf = _splice(compressed_buf, 0, n_compressed, kv.astype(compressed_buf.dtype))
    else:
        cslot = start_pos // ratio
        compressed_buf = _splice(
            compressed_buf, cslot, cslot + 1,
            mx.expand_dims(kv[:, 0], axis=1).astype(compressed_buf.dtype),
        )
    setattr(cache, slot_compressed, compressed_buf)

    return kv


class CSACompressor(nn.Module):
    """Per-layer CSA / HCA stream compressor. Forward = port of
    inference/model.py::Compressor with Phase 1 simplifications (see
    `_compressor_step` docstring).
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.compress_rope_theta = args.compress_rope_theta
        self.overlap = compress_ratio == 4
        # Phase 1 cache-buffer cap (avoid 1M-position eager allocation per layer).
        self.max_compressed_positions = max(1, args.compressed_max_seq_len // compress_ratio)
        coff = 2 if self.overlap else 1
        self.wgate = nn.QuantizedLinear(
            args.hidden_size, coff * self.head_dim, bits=8, group_size=args.group_size, bias=False
        )
        self.wkv = nn.QuantizedLinear(
            args.hidden_size, coff * self.head_dim, bits=8, group_size=args.group_size, bias=False
        )
        self.norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.ape = mx.zeros((compress_ratio, coff * self.head_dim), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, start_pos: int, cache) -> Optional[mx.array]:
        return _compressor_step(
            self, x, start_pos, cache,
            slot_compressed="compressed",
            slot_kv_state="compressor_kv_state",
            slot_score_state="compressor_score_state",
        )


class _IndexerCompressor(nn.Module):
    """Smaller compressor inside Indexer (compress_ratio==4 always, rotate=True
    in the reference; rotation skipped in Phase 1). On-disk weight names:
    `indexer.compressor.{wgate, wkv, norm, ape}`.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.compress_ratio = 4
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.compress_rope_theta = args.compress_rope_theta
        self.overlap = True
        self.max_compressed_positions = max(1, args.compressed_max_seq_len // self.compress_ratio)
        coff = 2  # always overlap for indexer
        self.wgate = nn.QuantizedLinear(
            args.hidden_size, coff * args.index_head_dim, bits=8, group_size=args.group_size, bias=False
        )
        self.wkv = nn.QuantizedLinear(
            args.hidden_size, coff * args.index_head_dim, bits=8, group_size=args.group_size, bias=False
        )
        self.norm = nn.RMSNorm(args.index_head_dim, eps=args.rms_norm_eps)
        self.ape = mx.zeros((self.compress_ratio, coff * args.index_head_dim), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, start_pos: int, cache) -> Optional[mx.array]:
        return _compressor_step(
            self, x, start_pos, cache,
            slot_compressed="indexer_compressed",
            slot_kv_state="indexer_compressor_kv_state",
            slot_score_state="indexer_compressor_score_state",
        )


class Indexer(nn.Module):
    """V4 sparse-topk indexer. Constructed only on layers with compress_ratio == 4.

    Tensor names match weights: `indexer.{wq_b, weights_proj, compressor.*}`.
    Forward = port of inference/model.py::Indexer.forward (model.py:402-433)
    with Phase 1 simplifications: skip q rotate_activation (line 414) and
    skip fp4_act_quant on q (line 416). Both rotations (q + Indexer's
    internal Compressor's rotate=True) cancel in the index_score einsum
    (Walsh-Hadamard orthogonal); skipping both preserves the einsum exactly.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads          # 64
        self.head_dim = args.index_head_dim        # 128
        self.rope_head_dim = args.qk_rope_head_dim # 64
        self.compress_rope_theta = args.compress_rope_theta
        self.compress_ratio = 4                    # only constructed on CSA layers
        self.index_topk = args.index_topk          # 512
        self.softmax_scale = self.head_dim ** -0.5
        self.wq_b = nn.QuantizedLinear(
            args.q_lora_rank, self.n_heads * self.head_dim,
            bits=8, group_size=args.group_size, bias=False,
        )
        self.weights_proj = nn.QuantizedLinear(
            args.hidden_size, self.n_heads,
            bits=8, group_size=args.group_size, bias=False,
        )
        self.compressor = _IndexerCompressor(args)

    def __call__(
        self,
        x: mx.array,                     # (B, S, hidden)
        qr: mx.array,                    # (B, S, q_lora_rank)
        start_pos: int,
        offset: int,
        cache,                           # V4Cache
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        # q path: wq_b -> reshape -> rope on rope-tail.
        q = self.wq_b(qr)                                              # (B, S, H*D)
        q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)        # (B, S, H, D)
        positions = mx.arange(start_pos, start_pos + seqlen, dtype=mx.float32)
        freqs_cis = _compressed_freqs_cis(positions, rd, self.compress_rope_theta)
        q_rope = apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = mx.concatenate([q[..., :-rd], q_rope.astype(q.dtype)], axis=-1)
        # SKIP Phase 1: rotate_activation(q), fp4_act_quant(q, ...).

        # Drive the internal compressor (writes to cache.indexer_compressed et al).
        self.compressor(x, start_pos, cache)

        # weights: (B, S, H) — per-token, per-head weight scalar.
        weights = self.weights_proj(x) * (self.softmax_scale * (self.n_heads ** -0.5))

        # index_score = einsum(q, indexer_compressed[:, :end_pos//ratio]).
        n_compressed = end_pos // ratio
        if n_compressed == 0:
            # Nothing compressed yet — return empty top-k.
            return mx.zeros((bsz, seqlen, 0), dtype=mx.int32)
        kv_indexed = cache.indexer_compressed[:bsz, :n_compressed]      # (B, T, D)
        index_score = mx.einsum("bshd,btd->bsht", q, kv_indexed)        # (B, S, H, T)
        # relu + per-head weight + reduce-over-heads -> (B, S, T)
        index_score = (mx.maximum(index_score, 0) * weights[..., None]).sum(axis=2)

        # Per-row future-position mask (model.py:424-426 generalized to
        # arbitrary start_pos). Fires whenever S > 1 — i.e. first prefill
        # (start_pos == 0) AND verify (start_pos > 0). For S == 1 (decode)
        # there are no future positions to mask: n_compressed = end_pos //
        # ratio already excludes anything past the current token, so the
        # mask would be all-False.
        if seqlen > 1:
            i_arr = mx.arange(
                start_pos + 1, start_pos + seqlen + 1, dtype=mx.int32,
            ).reshape(seqlen, 1)
            j_arr = mx.arange(n_compressed, dtype=mx.int32).reshape(1, n_compressed)
            mask = j_arr >= (i_arr // ratio)                            # (S, T) bool
            index_score = mx.where(mask[None, ...], -mx.inf, index_score)

        # Top-k via argsort (descending). Order doesn't matter to sparse_attn.
        k = min(self.index_topk, n_compressed)
        sorted_idxs = mx.argsort(-index_score, axis=-1)                 # (B, S, T)
        topk_idxs = sorted_idxs[..., :k].astype(mx.int32)               # (B, S, k)

        # Add offset; mask future positions whenever S > 1 (model.py:428-432
        # generalized — same per-row threshold as above).
        if seqlen > 1:
            i_arr = mx.arange(
                start_pos + 1, start_pos + seqlen + 1, dtype=mx.int32,
            ).reshape(1, seqlen, 1)
            future = topk_idxs >= (i_arr // ratio)
            topk_idxs = mx.where(future, mx.array(-1, dtype=mx.int32), topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


# ---------------------------------------------------------------------------
# Attention — MLA with grouped wo_a/wo_b
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """V4 MLA attention block.

    Forward (port of inference/model.py::Attention.forward):
      qr = q_norm(wq_a(x))                              # (B, S, q_lora_rank)
      q  = wq_b(qr).reshape(B, S, n_heads, head_dim)    # (B, S, 64, 512)
      q  = rsqrt-normalize per head (no learned weight)
      apply rope on last qk_rope_head_dim=64 of each head's head_dim
      kv = kv_norm(wkv(x)).reshape(B, S, 1, head_dim)
      apply rope on last 64 of kv
      cache.update_and_fetch(kv, kv)  # V4: keys == values
      attention with attn_sink (per-head scalar logit added to softmax denominator)
      o.shape (B, S, n_heads, head_dim) → reshape (B, S, n_groups=8, n_heads*head_dim/n_groups=4096)
      grouped einsum with wo_a viewed as (n_groups, o_lora_rank, in_per_group)
      flatten to (B, S, n_groups*o_lora_rank=8192) and apply wo_b → (B, S, hidden)
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.num_heads = args.num_attention_heads          # 64
        self.head_dim = args.head_dim                       # 512
        self.qk_rope_head_dim = args.qk_rope_head_dim       # 64
        self.q_lora_rank = args.q_lora_rank                 # 1024
        self.o_lora_rank = args.o_lora_rank                 # 1024
        self.o_groups = args.o_groups                       # 8
        self.in_per_group = args.num_attention_heads * args.head_dim // args.o_groups  # 4096
        self.scale = args.head_dim ** -0.5
        self.compress_ratio = args.compress_ratios[layer_idx]
        self.eps = args.rms_norm_eps

        # Q LoRA: hidden → q_lora_rank → num_heads*head_dim
        self.wq_a = nn.QuantizedLinear(
            args.hidden_size, args.q_lora_rank,
            bits=8, group_size=args.group_size, bias=False,
        )
        self.q_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.QuantizedLinear(
            args.q_lora_rank, args.num_attention_heads * args.head_dim,
            bits=8, group_size=args.group_size, bias=False,
        )

        # KV: hidden → head_dim (single KV head, unified head_dim)
        self.wkv = nn.QuantizedLinear(
            args.hidden_size, args.head_dim,
            bits=8, group_size=args.group_size, bias=False,
        )
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)

        # Output projection — wo_a is dequantized at sanitize time and held as a
        # plain (n_groups, o_lora_rank, in_per_group) tensor for grouped einsum.
        # wo_b is a regular (n_groups*o_lora_rank → hidden) QuantizedLinear.
        self.wo_a = mx.zeros(
            (args.o_groups, args.o_lora_rank, self.in_per_group), dtype=mx.bfloat16
        )
        self.wo_b = nn.QuantizedLinear(
            args.o_groups * args.o_lora_rank, args.hidden_size,
            bits=8, group_size=args.group_size, bias=False,
        )

        # Attention sinks — one per head, fp32.
        self.attn_sink = mx.zeros((args.num_attention_heads,), dtype=mx.bfloat16)

        # Two RoPE objects (standard + compressed-stream). traditional=True
        # selects the INTERLEAVED pair layout (pair k is dims (2k, 2k+1)),
        # matching the PyTorch reference at model.py:232-244 which uses
        # `unflatten(-1, (-1, 2))` + `view_as_complex`. The original port
        # used traditional=False (SPLIT layout: first half real / second
        # half imaginary, the Llama convention) which is INCOMPATIBLE with
        # the on-disk JANGTQ weights — those were trained against the
        # reference's interleaved rotation. Probed empirically 2026-05-06.
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            scaling_config=args.rope_parameters,
            max_position_embeddings=args.max_position_embeddings,
        )
        self.rope_compressed = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.compress_rope_theta,
            traditional=True,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )

        self.compressor: Optional[CSACompressor] = None
        self.indexer: Optional[Indexer] = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[V4Cache],
    ) -> mx.array:
        """Forward (port of inference/model.py::Attention.forward, model.py:484-543).

        Phase 1 wiring of Compressor + (optional) Indexer + sparse_attn into the
        attention path. Layout choice:
          - q, kv use self.rope_compressed when compress_ratio>0 (matches reference's
            per-layer freqs_cis at compress_rope_theta).
          - When compress_ratio==0 (layers 0, 1, 42 in this config), q/kv use a
            plain interleaved rope at args.rope_theta with NO YaRN, computed
            via _compressed_freqs_cis + apply_rotary_emb. This matches the
            reference at model.py:478-481 ("disable YaRN and use base rope_theta
            in pure sliding-window attention"). Phase 1.5 fix: layers 0-1
            sparse_attn + inverse-rope (forward direction must use the same
            freqs we invert with).
          - V4Cache holds the running (B, 1, T, hd) window-side KV (unbounded growth
            in Stage 1; Phase 2 swaps in the bounded circular buffer per the V4Cache
            structure node). cache.compressed holds the compressed-stream KV
            written by Compressor; the Indexer's internal compressor uses
            cache.indexer_compressed.
          - For prefill, sparse_attn consumes kv = concat(fresh_kv, compressed_kv).
            For decode, sparse_attn consumes kv = concat(V4Cache window slice,
            cache.compressed[:, :n_compressed]). topk_idxs concatenates the
            window indices and (compress_ratio>0 only) the indexer- or
            fallback-derived compressed indices.
          - Inverse RoPE on output applied for ALL layers (model.py:534, applied
            unconditionally in the reference). For compress_ratio>0 the freqs
            use compress_rope_theta; for compress_ratio==0 they use rope_theta
            (no YaRN). Phase 1.5 fix: layers 0-1 sparse_attn + inverse-rope.
        """
        B, S, _ = x.shape
        nh, hd, rd = self.num_heads, self.head_dim, self.qk_rope_head_dim
        a = self.args

        offset = cache.offset if cache is not None else 0

        # Forward freqs_cis selection (matches reference model.py:475-481).
        # compress_ratio>0  : self.rope_compressed (mlx_lm RoPE, traditional=True,
        #                     base=compress_rope_theta, no YaRN).
        # compress_ratio==0 : _compressed_freqs_cis at args.rope_theta, no YaRN.
        # Phase 1.5 fix: layers 0-1 use plain rope (no YaRN) so that the inverse
        # rope on the attention output below correctly cancels the forward.
        if self.compress_ratio > 0:
            fwd_freqs_cis = None  # using mlx_lm rope_obj path below
            rope_obj = self.rope_compressed
        else:
            positions = mx.arange(offset, offset + S, dtype=mx.float32)
            fwd_freqs_cis = _compressed_freqs_cis(positions, rd, a.rope_theta)
            rope_obj = None

        # Q path ---------------------------------------------------------------
        qr = self.q_norm(self.wq_a(x))                                         # (B, S, q_lora_rank)
        q = self.wq_b(qr).reshape(B, S, nh, hd)                                # (B, S, nh, hd)
        q = q * mx.rsqrt(q.square().mean(-1, keepdims=True) + self.eps)
        if rope_obj is not None:
            q_rope = q[..., -rd:].transpose(0, 2, 1, 3)                        # (B, nh, S, rd)
            q_rope = rope_obj(q_rope, offset=offset)
            q = mx.concatenate([q[..., :hd - rd], q_rope.transpose(0, 2, 1, 3)], axis=-1)
        else:
            # Phase 1.5 fix: layers 0-1 sparse_attn + inverse-rope (forward).
            q_rope = apply_rotary_emb(q[..., -rd:], fwd_freqs_cis)             # (B, S, nh, rd)
            q = mx.concatenate([q[..., :hd - rd], q_rope.astype(q.dtype)], axis=-1)
        # q stays as (B, S, nh, hd) — sparse_attn's expected layout.

        # KV path (single KV head) ---------------------------------------------
        kv = self.kv_norm(self.wkv(x)).reshape(B, S, 1, hd)                    # (B, S, 1, hd)
        if rope_obj is not None:
            kv_rope = kv[..., -rd:].transpose(0, 2, 1, 3)                      # (B, 1, S, rd)
            kv_rope = rope_obj(kv_rope, offset=offset)
            kv = mx.concatenate([kv[..., :hd - rd], kv_rope.transpose(0, 2, 1, 3)], axis=-1)
        else:
            # Phase 1.5 fix: layers 0-1 sparse_attn + inverse-rope (forward).
            kv_rope = apply_rotary_emb(kv[..., -rd:], fwd_freqs_cis)           # (B, S, 1, rd)
            kv = mx.concatenate([kv[..., :hd - rd], kv_rope.astype(kv.dtype)], axis=-1)

        # V4Cache update (always — even for compress_ratio==0 layers, so future
        # decode steps have a window history). Cache stores (B, 1, T, hd).
        kv_for_cache = kv.transpose(0, 2, 1, 3)                                # (B, 1, S, hd)
        if cache is not None:
            keys_cache, _ = cache.update_and_fetch(kv_for_cache, kv_for_cache)  # (B, 1, T_window, hd)
        else:
            keys_cache = kv_for_cache

        # Mode disambiguation:
        #   is_first_prefill : S > 1 AND offset == 0 — fresh prefill, no prior cache.
        #   is_verify        : S > 1 AND offset > 0  — speculative decoding's verify
        #                      pass (third regime). kv_for_sparse uses the V4Cache
        #                      window slice plus cache.compressed[:n_compressed]
        #                      (post-third-regime emissions written), with per-row
        #                      window topk for the S queries.
        #   else             : S == 1 — decode (single-token append).
        is_first_prefill = (S > 1 and offset == 0)
        is_verify = (S > 1 and offset > 0)
        win = a.sliding_window

        # Compressor path (compress_ratio>0 layers only) -----------------------
        # Drives the gated-pool state machine; writes any new compressed kv to
        # cache.compressed. Returns the freshly-emitted slice (or None mid-cycle).
        # In verify mode the compressor unrolls S decode steps (see
        # _compressor_step's third regime).
        compressed_kv = None
        if self.compress_ratio > 0 and self.compressor is not None and cache is not None:
            compressed_kv = self.compressor(x, offset, cache)

        # Build kv_for_sparse and topk_idxs ------------------------------------
        if is_first_prefill:
            # Fresh kv (B, S, hd) — squeeze the n_kv=1 axis; sparse_attn wants
            # (B, N, hd) with no head dim.
            fresh_kv = kv.squeeze(2)                                            # (B, S, hd)
            if compressed_kv is not None:
                kv_for_sparse = mx.concatenate([fresh_kv, compressed_kv], axis=1)
                n_compressed = compressed_kv.shape[1]
            else:
                kv_for_sparse = fresh_kv
                n_compressed = 0

            # Window topk: per-query upper-triangular indices into [0, S).
            win_topk = get_window_topk_idxs(win, B, S, 0)                       # (B, S, win)

            if self.compress_ratio > 0 and n_compressed > 0:
                # Compressed entries live at indices [S, S+n_compressed) in kv_for_sparse.
                offset_cmp = S
                if self.indexer is not None:
                    # offset==0 here; passing it explicitly keeps the indexer's
                    # start_pos tied to the cache state (matches inference
                    # reference at model.py:511).
                    cmp_topk = self.indexer(x, qr, offset, offset_cmp, cache)   # (B, S, k)
                else:
                    cmp_topk = get_compress_topk_idxs(
                        self.compress_ratio, B, S, 0, offset_cmp,
                    )                                                            # (B, S, n_compressed)
                topk_idxs = mx.concatenate([win_topk, cmp_topk], axis=-1)
            else:
                topk_idxs = win_topk
        elif is_verify:
            # Verify (third regime): S > 1 onto a non-empty cache. kv_for_sparse
            # = V4Cache window slice + cache.compressed[:n_compressed] (which
            # now includes any emissions written by the unrolled compressor
            # above). win_topk is per-row causal: row i (query at absolute
            # position offset+i) sees window slots [0, T_window-S+i].
            win_kv = keys_cache.squeeze(1)                                      # (B, T_window, hd)
            T_window = win_kv.shape[1]

            if (
                self.compress_ratio > 0
                and cache is not None
                and cache.compressed is not None
            ):
                # All emissions through end_pos are in cache.compressed.
                n_compressed = (offset + S) // self.compress_ratio
            else:
                n_compressed = 0

            if n_compressed > 0:
                kv_for_sparse = mx.concatenate(
                    [win_kv, cache.compressed[:, :n_compressed]], axis=1,
                )
            else:
                kv_for_sparse = win_kv

            # Per-row windowed-causal topk. The new S tokens occupy the last S
            # slots in T_window (V4Cache appends chronologically); row i is at
            # T_window index (T_window - S + i), which is absolute position
            # (offset + i). Windowed-causal: row i sees T_window indices
            # [max(0, offset+i-win+1), offset+i] — same set as monolithic
            # prefill's get_window_topk_idxs(win, B, S, 0) at the equivalent
            # absolute positions. Without the lower bound, chunked prefill of
            # sequences longer than `win` feeds the model attention contexts
            # wider than what it was trained on, producing coherence collapse.
            positions = mx.arange(T_window, dtype=mx.int32).reshape(1, T_window)
            limits = mx.arange(
                T_window - S, T_window, dtype=mx.int32,
            ).reshape(S, 1)
            lower = mx.maximum(
                limits - win + 1, mx.array(0, dtype=mx.int32),
            )
            row_mask = (positions >= lower) & (positions <= limits)
            win_topk_local = mx.where(
                row_mask, positions, mx.array(-1, dtype=mx.int32),
            )                                                                    # (S, T_window)
            win_topk = mx.broadcast_to(
                win_topk_local[None, ...], (B, S, T_window),
            )                                                                    # (B, S, T_window)

            if n_compressed > 0:
                offset_cmp = T_window
                if self.indexer is not None:
                    cmp_topk = self.indexer(x, qr, offset, offset_cmp, cache)
                else:
                    cmp_topk = get_compress_topk_idxs(
                        self.compress_ratio, B, S, offset, offset_cmp,
                    )
                topk_idxs = mx.concatenate([win_topk, cmp_topk], axis=-1)
            else:
                topk_idxs = win_topk
        else:
            # Decode (S == 1): kv = V4Cache window slice +
            # cache.compressed[:n_compressed]. V4Cache returns (B, 1, T_window,
            # hd); squeeze the n_kv axis.
            win_kv = keys_cache.squeeze(1)                                      # (B, T_window, hd)
            T_window = win_kv.shape[1]

            if (
                self.compress_ratio > 0
                and cache is not None
                and cache.compressed is not None
            ):
                n_compressed = (offset + 1) // self.compress_ratio              # ref model.py:271
            else:
                n_compressed = 0

            if n_compressed > 0:
                kv_for_sparse = mx.concatenate(
                    [win_kv, cache.compressed[:, :n_compressed]], axis=1,
                )
            else:
                kv_for_sparse = win_kv

            # Window topk for decode: full attention to the V4Cache linear slice.
            # (Reference uses circular indexing into a fixed-size buffer; our V4Cache
            # returns chronological [oldest..newest] last-T_window entries, so the
            # equivalent set of attended positions is just [0, T_window).)
            win_topk = mx.broadcast_to(
                mx.arange(T_window, dtype=mx.int32).reshape(1, 1, T_window),
                (B, 1, T_window),
            )

            if n_compressed > 0:
                offset_cmp = T_window
                if self.indexer is not None:
                    cmp_topk = self.indexer(x, qr, offset, offset_cmp, cache)
                else:
                    cmp_topk = get_compress_topk_idxs(
                        self.compress_ratio, B, S, offset, offset_cmp,
                    )
                topk_idxs = mx.concatenate([win_topk, cmp_topk], axis=-1)
            else:
                topk_idxs = win_topk

        # sparse_attn ----------------------------------------------------------
        # q: (B, S, nh, hd), kv_for_sparse: (B, N, hd), attn_sink: (nh,),
        # topk_idxs: (B, S, K), scale: float. Output: (B, S, nh, hd).
        o = sparse_attn(q, kv_for_sparse, self.attn_sink, topk_idxs, self.scale)

        # Inverse RoPE on output's rope tail (model.py:534, applied for ALL
        # layers in the reference). Phase 1.5 fix: layers 0-1 sparse_attn +
        # inverse-rope (was previously skipped on compress_ratio==0 layers).
        # The inverse freqs MUST match the forward freqs used for q/kv above.
        positions = mx.arange(offset, offset + S, dtype=mx.float32)
        if self.compress_ratio > 0:
            inv_freqs = _compressed_freqs_cis(positions, rd, a.compress_rope_theta)
        else:
            inv_freqs = _compressed_freqs_cis(positions, rd, a.rope_theta)
        o_rope_tail = apply_rotary_emb(o[..., -rd:], inv_freqs, inverse=True)
        o = mx.concatenate([o[..., :hd - rd], o_rope_tail.astype(o.dtype)], axis=-1)

        # Grouped output projection (unchanged) --------------------------------
        o_grouped = o.reshape(B, S, self.o_groups, self.in_per_group)
        out = mx.einsum("bsgd,grd->bsgr", o_grouped, self.wo_a)
        out = out.reshape(B, S, self.o_groups * self.o_lora_rank)
        return self.wo_b(out)


# ---------------------------------------------------------------------------
# Shared expert MLP (with swiglu clipping)
# ---------------------------------------------------------------------------


class DeepseekV4MLP(nn.Module):
    """SwiGLU MLP with optional clipping (V4 uses swiglu_limit=10.0)."""

    def __init__(self, args: ModelArgs, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.swiglu_limit = args.swiglu_limit
        self.w1 = nn.QuantizedLinear(
            hidden_size, intermediate_size, bits=8, group_size=args.group_size, bias=False
        )
        self.w2 = nn.QuantizedLinear(
            intermediate_size, hidden_size, bits=8, group_size=args.group_size, bias=False
        )
        self.w3 = nn.QuantizedLinear(
            hidden_size, intermediate_size, bits=8, group_size=args.group_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        return self.w2(nn.silu(gate) * up)


# ---------------------------------------------------------------------------
# MoE — gate (sqrtsoftplus + tid2eid override) + SwitchGLU + shared expert
# ---------------------------------------------------------------------------


class MoEGate(nn.Module):
    """V4 router. First `n_hash_layers` layers use static `tid2eid` lookup (no
    scoring). Remaining layers use sqrtsoftplus(scores) + bias for top-k.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.is_hash = layer_idx < args.num_hash_layers
        # Router weight is fp16 (kept high precision per JANGTQ norms_router_hc=16).
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size), dtype=mx.bfloat16)
        if self.is_hash:
            # tid2eid: per-vocab → top-k expert ids. I64.
            self.tid2eid = mx.zeros((args.vocab_size, args.num_experts_per_tok), dtype=mx.int32)
        else:
            # e_score_correction_bias for non-hash layers.
            self.bias = mx.zeros((args.n_routed_experts,), dtype=mx.bfloat16)

    def __call__(
        self, x: mx.array, token_ids: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        # x: (B, S, hidden) flattened by caller to (N, hidden) for the gate.
        # Returns (indices, weights) for top-k expert dispatch.
        scores = mx.matmul(x.astype(mx.float32), self.weight.T.astype(mx.float32))
        # sqrtsoftplus
        scores = mx.sqrt(nn.softplus(scores))
        original_scores = scores
        if not self.is_hash:
            scores = scores + self.bias
        if self.is_hash:
            assert token_ids is not None, "tid2eid requires token_ids"
            # tid2eid lookup: (B, S) -> (B, S, top_k)
            indices = self.tid2eid[token_ids]
        else:
            indices = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        weights = mx.take_along_axis(original_scores, indices, axis=-1)
        # sqrtsoftplus path always normalizes.
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)
        weights = weights * self.routed_scaling_factor
        return indices.astype(mx.uint32), weights


class ClippedSwiGLU(nn.Module):
    """SwiGLU with clipping (matches DeepSeek V4 reference Expert behavior)."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        if self.limit > 0:
            x_up = mx.clip(x_up, -self.limit, self.limit)
            x_gate = mx.minimum(x_gate, self.limit)
        return nn.silu(x_gate) * x_up


class DeepseekV4MoE(nn.Module):
    """V4 MoE block. Routed experts use the custom JANGTQ 2-bit MXTQ kernel
    (with on-the-fly Hadamard-on-input rotation); shared expert is a regular
    8-bit affine SwiGLU MLP.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.gate = MoEGate(args, layer_idx)
        self.routed_experts = JANGTQRoutedExperts(args)
        self.shared_experts = DeepseekV4MLP(args, args.hidden_size, args.moe_intermediate_size)

    def __call__(
        self,
        x: mx.array,
        token_ids: Optional[mx.array] = None,
        jangtq: Optional[dict[str, mx.array]] = None,
    ) -> mx.array:
        assert jangtq is not None, "DeepseekV4MoE requires JANGTQ codebook + signs"
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        tids_flat = token_ids.reshape(-1) if token_ids is not None else None
        inds, scores = self.gate(x_flat, token_ids=tids_flat)
        y = self.routed_experts(x_flat, inds, scores, jangtq)   # (N, D)
        y = y.reshape(B, S, D)
        return y + self.shared_experts(x)


# ---------------------------------------------------------------------------
# Decoder layer — Hyper-Connections residual mixing
# ---------------------------------------------------------------------------


@mx.compile
def hc_pre(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    """Collapse HC copies (B, S, hc, dim) → (B, S, dim) via Sinkhorn pre-weights."""
    B, S, hc, D = x.shape
    x_flat = x.reshape(B, S, hc * D).astype(mx.float32)
    rsqrt = mx.rsqrt(x_flat.square().mean(-1, keepdims=True) + norm_eps)
    mixes = mx.matmul(x_flat, hc_fn.T) * rsqrt              # (B, S, mix_hc)
    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
    )
    # y[b,s,d] = sum_h pre[b,s,h] * x[b,s,h,d]
    y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
    return y.astype(x.dtype), post, comb


@mx.compile
def hc_post(
    x: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
):
    """Expand (B, S, dim) → (B, S, hc, dim) using post-weights and combination matrix.

    y[b,s,h,d] = post[b,s,h] * x[b,s,d] + sum_k comb[b,s,h,k] * residual[b,s,k,d]
    """
    a = post[..., None] * x[..., None, :]                  # (B, S, hc, dim)
    b = mx.einsum("bshk,bskd->bshd", comb.astype(residual.dtype), residual)
    return (a + b).astype(x.dtype)


def hc_head(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    norm_eps: float,
    hc_eps: float,
):
    """Final HC collapse before lm_head — sigmoid (not Sinkhorn).

    x: (B, S, hc, dim)
    hc_fn: (hc_mult, hc_mult * dim)
    hc_scale: (1,), hc_base: (hc_mult,)
    """
    B, S, hc, D = x.shape
    x_flat = x.reshape(B, S, hc * D).astype(mx.float32)
    rsqrt = mx.rsqrt(x_flat.square().mean(-1, keepdims=True) + norm_eps)
    mixes = mx.matmul(x_flat, hc_fn.T) * rsqrt              # (B, S, hc)
    pre = mx.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
    return y.astype(x.dtype)


class DeepseekV4DecoderLayer(nn.Module):
    """One V4 decoder block. Hidden state carried as (B, S, hc_mult, dim)."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.compress_ratio = args.compress_ratios[layer_idx]

        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = DeepseekV4Attention(args, layer_idx)
        if self.compress_ratio > 0:
            self.attn.compressor = CSACompressor(args, self.compress_ratio)
            # Indexer is constructed only on compress_ratio == 4 layers (per ref).
            if self.compress_ratio == 4:
                self.attn.indexer = Indexer(args)

        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = DeepseekV4MoE(args, layer_idx)

        # mHC (HC) parameters per the reference's Block.__init__:
        #   mix_hc = (2 + hc_mult) * hc_mult        (= 24 for hc_mult=4)
        #   hc_dim = hc_mult * hidden_size          (= 16384)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.bfloat16)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.bfloat16)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.bfloat16)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.bfloat16)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.bfloat16)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.bfloat16)

    def __call__(
        self,
        x: mx.array,                       # (B, S, hc_mult, hidden)
        mask: Optional[mx.array],
        cache: Optional[V4Cache],
        token_ids: Optional[mx.array] = None,
        jangtq: Optional[dict[str, mx.array]] = None,
    ) -> mx.array:
        a = self.args

        # Attention sub-block ---------------------------------------------------
        residual = x
        h, post, comb = hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            a.hc_mult, a.hc_sinkhorn_iters, a.rms_norm_eps, a.hc_eps,
        )
        h = self.attn_norm(h)
        h = self.attn(h, mask, cache)
        x = hc_post(h, residual, post, comb)

        # FFN sub-block --------------------------------------------------------
        residual = x
        h, post, comb = hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            a.hc_mult, a.hc_sinkhorn_iters, a.rms_norm_eps, a.hc_eps,
        )
        h = self.ffn_norm(h)
        h = self.ffn(h, token_ids=token_ids, jangtq=jangtq)
        x = hc_post(h, residual, post, comb)
        return x


# ---------------------------------------------------------------------------
# MTP-native speculative drafter — skeleton for Option 3 hybrid bolt-on.
#
# Step 1 scope: structural attributes match the bf16 weight layout produced
# by `_load_mtp_weights.py` so `model.load_weights` can populate them. The
# forward pass is intentionally a stub (raises NotImplementedError) — that
# work lands in step 2 (MTP forward + speculative validator loop).
#
# Why custom classes parallel to DeepseekV4Attention/MoE:
#   * The main model's attention/MoE use JANGTQ 8-bit affine + 2-bit MXTQ
#     dispatch. The non-JANGTQ MTP weights are bf16-dequanted at load time
#     and use plain `nn.Linear`. Reusing the JANGTQ classes would force
#     either re-quantization (recipe opaque, see `MTP weights source` node)
#     or a load-time conversion path. Plain bf16 is simpler for first-light.
#   * Drafter mismatch (non-JANGTQ-trained MTP block running with
#     JANGTQ-quantized embed/head) only affects ACCEPTANCE RATE, never
#     output quality — validator runs the full main model on every token.
#     See SiftText `Speculative decoding output guarantee` node.
# ---------------------------------------------------------------------------


class MTPMoEGate(nn.Module):
    """MTP MoE gate. Layer 43 is past num_hash_layers=3, so this is the
    sqrtsoftplus + bias path. Bias is fp32 in non-JANGTQ source vs bf16 in
    JANGTQ; we keep fp32 to match the on-disk dtype.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size), dtype=mx.bfloat16)
        self.bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scores = mx.matmul(x.astype(mx.float32), self.weight.T.astype(mx.float32))
        scores = mx.sqrt(nn.softplus(scores))
        original_scores = scores
        scores = scores + self.bias
        indices = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        weights = mx.take_along_axis(original_scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)
        weights = weights * self.routed_scaling_factor
        return indices.astype(mx.uint32), weights


class MTPRoutedExperts(nn.Module):
    """BF16 routed experts for the MTP block. No JANGTQ dispatch, no
    Hadamard rotation — weights pre-dequanted at load time to plain bf16.
    Forward is plain gather + matmul.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts = args.n_routed_experts
        self.hidden = args.hidden_size
        self.inter = args.moe_intermediate_size
        self.swiglu_limit = args.swiglu_limit
        self.w1 = mx.zeros((self.n_experts, self.inter, self.hidden), dtype=mx.bfloat16)
        self.w2 = mx.zeros((self.n_experts, self.hidden, self.inter), dtype=mx.bfloat16)
        self.w3 = mx.zeros((self.n_experts, self.inter, self.hidden), dtype=mx.bfloat16)

    def __call__(
        self, x_flat: mx.array, indices: mx.array, weights: mx.array,
    ) -> mx.array:
        # Plain bf16 gather + einsum. The transient (N, K, I, H) tensors
        # bound first-light memory at small N (decode validates K draft
        # positions). If real-prefill N grows large, chunk like
        # JANGTQRoutedExperts.CHUNK_N.
        w1_sel = self.w1[indices]
        w3_sel = self.w3[indices]
        w2_sel = self.w2[indices]
        gate = mx.einsum("nkih,nh->nki", w1_sel, x_flat.astype(w1_sel.dtype))
        up = mx.einsum("nkih,nh->nki", w3_sel, x_flat.astype(w3_sel.dtype))
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        act = nn.silu(gate) * up
        out = mx.einsum("nkhi,nki->nkh", w2_sel, act)
        return (out * weights[..., None].astype(out.dtype)).sum(axis=1)


class MTPMLP(nn.Module):
    """BF16 SwiGLU MLP for the MTP shared expert."""

    def __init__(self, args: ModelArgs, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.swiglu_limit = args.swiglu_limit
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        return self.w2(nn.silu(gate) * up)


class MTPMoE(nn.Module):
    """BF16 MoE for the MTP block."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MTPMoEGate(args)
        self.routed_experts = MTPRoutedExperts(args)
        self.shared_experts = MTPMLP(args, args.hidden_size, args.moe_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        inds, scores = self.gate(x_flat)
        y = self.routed_experts(x_flat, inds, scores)
        y = y.reshape(B, S, D)
        return y + self.shared_experts(x)


class MTPAttention(nn.Module):
    """BF16 attention for the MTP block. Same shapes as
    DeepseekV4Attention but with non-quantized linear layers. compress_ratio
    is 0 for the MTP layer (per non-JANGTQ config compress_ratios[-1]==0)
    so no Compressor / Indexer / sparse_attn — plain sliding-window SDPA.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        a = args
        self.wq_a = nn.Linear(a.hidden_size, a.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(a.q_lora_rank, eps=a.rms_norm_eps)
        self.wq_b = nn.Linear(
            a.q_lora_rank, a.num_attention_heads * a.head_dim, bias=False
        )
        self.wkv = nn.Linear(a.hidden_size, a.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(a.head_dim, eps=a.rms_norm_eps)
        # wo_a is a grouped LoRA (o_groups, o_lora_rank, in_per_group), not a
        # standard Linear. Stored as a raw mx.array so the loader can reshape
        # the on-disk flat (8192, 4096) bf16 directly into the grouped layout
        # the attention forward expects.
        in_per_group = a.num_attention_heads * a.head_dim // a.o_groups
        self.wo_a = mx.zeros(
            (a.o_groups, a.o_lora_rank, in_per_group), dtype=mx.bfloat16
        )
        self.wo_b = nn.Linear(a.o_groups * a.o_lora_rank, a.hidden_size, bias=False)
        # attn_sink: fp32 [n_heads], same role as DeepseekV4Attention.attn_sink.
        self.attn_sink = mx.zeros((a.num_attention_heads,), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[V4Cache],
    ) -> mx.array:
        # MTP layer is compress_ratio==0: plain sliding-window SDPA via
        # sparse_attn with attn_sink, no compressor / indexer. RoPE forward
        # and inverse use base rope_theta with no YaRN (matches the
        # layers-0/1/42 path in DeepseekV4Attention).
        a = self.args
        B, S, _ = x.shape
        nh, hd, rd = a.num_attention_heads, a.head_dim, a.qk_rope_head_dim
        in_per_group = nh * hd // a.o_groups
        scale = hd ** -0.5
        offset = cache.offset if cache is not None else 0

        positions = mx.arange(offset, offset + S, dtype=mx.float32)
        fwd_freqs_cis = _compressed_freqs_cis(positions, rd, a.rope_theta)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, S, nh, hd)
        q = q * mx.rsqrt(q.square().mean(-1, keepdims=True) + a.rms_norm_eps)
        q_rope = apply_rotary_emb(q[..., -rd:], fwd_freqs_cis)
        q = mx.concatenate([q[..., :hd - rd], q_rope.astype(q.dtype)], axis=-1)

        kv = self.kv_norm(self.wkv(x)).reshape(B, S, 1, hd)
        kv_rope = apply_rotary_emb(kv[..., -rd:], fwd_freqs_cis)
        kv = mx.concatenate([kv[..., :hd - rd], kv_rope.astype(kv.dtype)], axis=-1)

        kv_for_cache = kv.transpose(0, 2, 1, 3)
        if cache is not None:
            keys_cache, _ = cache.update_and_fetch(kv_for_cache, kv_for_cache)
        else:
            keys_cache = kv_for_cache

        is_prefill = (S > 1)
        win = a.sliding_window
        if is_prefill:
            kv_for_sparse = kv.squeeze(2)
            topk_idxs = get_window_topk_idxs(win, B, S, 0)
        else:
            win_kv = keys_cache.squeeze(1)
            T_window = win_kv.shape[1]
            kv_for_sparse = win_kv
            topk_idxs = mx.broadcast_to(
                mx.arange(T_window, dtype=mx.int32).reshape(1, 1, T_window),
                (B, 1, T_window),
            )

        o = sparse_attn(q, kv_for_sparse, self.attn_sink, topk_idxs, scale)

        inv_freqs = _compressed_freqs_cis(positions, rd, a.rope_theta)
        o_rope_tail = apply_rotary_emb(o[..., -rd:], inv_freqs, inverse=True)
        o = mx.concatenate([o[..., :hd - rd], o_rope_tail.astype(o.dtype)], axis=-1)

        o_grouped = o.reshape(B, S, a.o_groups, in_per_group)
        out = mx.einsum("bsgd,grd->bsgr", o_grouped, self.wo_a)
        out = out.reshape(B, S, a.o_groups * a.o_lora_rank)
        return self.wo_b(out)


class MTPBlock(nn.Module):
    """V4 Flash MTP block for speculative drafting (depth=1).

    Mirrors `inference/model.py::MTPBlock` (lines 738-766): inherits Block's
    submodules (attn, ffn, norms, HC residuals) plus MTP-specific
    e_proj / h_proj / enorm / hnorm / norm and hc_head_*. embed and head are
    SHARED with the main model post-construction (set as plain Python
    attributes after Model.__init__ assembles both).

    Step 1 scope: parameter shapes match the bf16 dict produced by
    `_load_mtp_weights.py`. Forward is a stub — see step 2 for the actual
    e_proj+h_proj merge → attn+MoE → head pipeline.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Block-equivalent submodules.
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = MTPAttention(args)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = MTPMoE(args)
        # MTP-specific submodules.
        self.e_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.h_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # HC parameters. Note non-JANGTQ stores these as fp32; the main
        # model's DeepseekV4DecoderLayer holds them as bf16 because JANGTQ
        # downcasts. Keep fp32 here to match the on-disk dtype.
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_head_fn = mx.zeros((args.hc_mult, hc_dim), dtype=mx.float32)
        self.hc_head_base = mx.zeros((args.hc_mult,), dtype=mx.float32)
        self.hc_head_scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,                       # (B, S, hc_mult, hidden) carry
        mask: Optional[mx.array],
        cache: Optional[V4Cache],
        input_ids: mx.array,               # (B, S)
    ) -> mx.array:
        logits, _ = self.forward_with_carry(x, mask, cache, input_ids)
        return logits

    def forward_with_carry(
        self,
        x: mx.array,                       # (B, S, hc_mult, hidden) carry
        mask: Optional[mx.array],
        cache: Optional[V4Cache],
        input_ids: mx.array,               # (B, S)
    ) -> tuple[mx.array, mx.array]:
        # Mirrors inference/model.py::MTPBlock.forward (lines 756-766) plus
        # an inlined Block.forward (since MTPBlock does not subclass
        # DeepseekV4DecoderLayer in the MLX engine). Returns (logits, carry)
        # where carry is the post-Block (B, S, hc_mult, hidden) hidden state
        # immediately before the final HC collapse + norm + shared head — the
        # value used as input `x` to the next MTP step when chaining K>1
        # drafts autoregressively in the speculative loop.
        a = self.args
        assert hasattr(self, "embed") and hasattr(self, "head"), \
            "MTPBlock requires shared embed/head set post-construction"

        # e_proj / h_proj wrap (ref model.py:760-763) -------------------------
        e = self.enorm(self.embed(input_ids))                     # (B, S, hidden)
        x = self.hnorm(x)                                         # (B, S, hc, hidden)
        x = mx.expand_dims(self.e_proj(e), 2) + self.h_proj(x)    # broadcast add

        # Inline Block.forward (mirrors DeepseekV4DecoderLayer.__call__) -----
        residual = x
        h, post, comb = hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            a.hc_mult, a.hc_sinkhorn_iters, a.rms_norm_eps, a.hc_eps,
        )
        h = self.attn_norm(h)
        h = self.attn(h, mask, cache)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            a.hc_mult, a.hc_sinkhorn_iters, a.rms_norm_eps, a.hc_eps,
        )
        h = self.ffn_norm(h)
        h = self.ffn(h)
        x = hc_post(h, residual, post, comb)

        carry = x  # (B, S, hc_mult, hidden), pre-collapse — chain target for K>1.

        # Final HC collapse + norm + shared head (ref model.py:765) -----------
        h = hc_head(
            carry, self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
            a.rms_norm_eps, a.hc_eps,
        )
        h = self.norm(h)
        return self.head(h), carry


# ---------------------------------------------------------------------------
# Top-level Model
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """DeepSeek V4 Flash. Module attribute names match weight keys (flat namespace).

      embed                                    QuantizedEmbedding
      layers.{0..42}.*                         DeepseekV4DecoderLayer
      norm                                     RMSNorm(hidden=4096)
      head                                     QuantizedLinear(hidden, vocab)
      hc_head_{base, fn, scale}                Final HC collapse (sigmoid form)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.embed = nn.QuantizedEmbedding(
            args.vocab_size, args.hidden_size, bits=8, group_size=args.group_size,
        )
        self.layers = [
            DeepseekV4DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.head = nn.QuantizedLinear(
            args.hidden_size, args.vocab_size,
            bits=8, group_size=args.group_size, bias=False,
        )

        # Final HC collapse (sigmoid form — see inference/model.py::ParallelHead.hc_head).
        # Shape: hc_fn (hc_mult, hc_mult*hidden), hc_base (hc_mult,), hc_scale (1,).
        self.hc_head_fn = mx.zeros(
            (args.hc_mult, args.hc_mult * args.hidden_size), dtype=mx.bfloat16
        )
        self.hc_head_base = mx.zeros((args.hc_mult,), dtype=mx.bfloat16)
        self.hc_head_scale = mx.zeros((1,), dtype=mx.bfloat16)

        # JANGTQ runtime constants — codebooks (4 fp32 entries each) and randomized
        # Hadamard sign vectors (±1 fp32, deterministic per input dim and seed=42).
        # Loaded from `jangtq_runtime.safetensors` via sanitize() — see module
        # docstring for the symlink that makes mlx_lm pick the sidecar up.
        self.jangtq_cb_4096 = mx.zeros((4,), dtype=mx.float32)
        self.jangtq_cb_2048 = mx.zeros((4,), dtype=mx.float32)
        self.jangtq_signs_4096 = mx.zeros((args.hidden_size,), dtype=mx.float32)
        self.jangtq_signs_2048 = mx.zeros((args.moe_intermediate_size,), dtype=mx.float32)

        # MTP-native speculative drafter (Option 3 hybrid bolt-on).
        # Constructed only when args.load_mtp; weights come from non-JANGTQ
        # Flash via sanitize(). embed / head are shared with the main model
        # (assigned post-construction as plain Python refs in step 2 wiring;
        # not declared here to avoid double-tracking the shared parameters).
        if args.load_mtp:
            self.mtp = MTPBlock(args)
            # Share embed and head with the main model. Use object.__setattr__
            # so the assignment bypasses nn.Module's child-registration path
            # — otherwise tree_flatten would walk these parameters twice and
            # load_weights would receive duplicate keys.
            object.__setattr__(self.mtp, "embed", self.embed)
            object.__setattr__(self.mtp, "head", self.head)

    def __call__(
        self,
        inputs: mx.array,                       # (B, S) int token ids
        cache: Optional[list[V4Cache]] = None,
    ) -> mx.array:
        logits, _ = self.forward_with_carry(inputs, cache=cache)
        return logits

    def forward_with_carry(
        self,
        inputs: mx.array,                       # (B, S) int token ids
        cache: Optional[list[V4Cache]] = None,
    ) -> tuple[mx.array, mx.array]:
        # Same forward as __call__, but additionally returns the pre-collapse
        # HC carry of shape (B, S, hc_mult, hidden) used as input `x` to
        # MTPBlock during speculative drafting. The carry is the value of `h`
        # right after the last decoder layer, before hc_head's final collapse.
        B, S = inputs.shape
        a = self.args

        h = self.embed(inputs)                  # (B, S, hidden)
        # Expand to hc_mult parallel hidden-state copies.
        h = mx.broadcast_to(h[:, :, None, :], (B, S, a.hc_mult, a.hidden_size))
        h = mx.array(h)                         # materialize (broadcast may be lazy)

        if cache is None:
            cache = [None] * len(self.layers)

        # Mask: window-aware causal. For Stage 1 prefill of short sequences, this
        # is the standard causal mask (window not yet hit).
        mask = None
        if S > 1:
            mask = create_attention_mask(
                h[..., 0, :], cache=cache[0],
                window_size=a.sliding_window, return_array=True,
            )

        jangtq = {
            "cb_4096": self.jangtq_cb_4096,
            "cb_2048": self.jangtq_cb_2048,
            "signs_4096": self.jangtq_signs_4096,
            "signs_2048": self.jangtq_signs_2048,
        }
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, token_ids=inputs, jangtq=jangtq)

        carry = h  # (B, S, hc_mult, hidden), pre-collapse — input shape for MTP.

        # Final HC collapse via sigmoid (not Sinkhorn) before norm + head.
        h = hc_head(
            carry, self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
            a.rms_norm_eps, a.hc_eps,
        )
        h = self.norm(h)
        return self.head(h), carry

    # -- mlx_lm Model interface --------------------------------------------

    def make_cache(self) -> list[V4Cache]:
        return [V4Cache(self.args, i) for i in range(self.args.num_hidden_layers)]

    def cast_predicate(self, key: str) -> bool:
        """Return True for keys that should stay at fp16 (i.e., NOT quantized).

        Per JANGTQ config (norms_router_hc=16): norms, router gate.weight,
        router gate.bias, mHC tensors, attn_sink, compressor.{ape, norm.weight},
        wo_a (held dequantized as plain bf16), and the routed-expert MXTQ
        tensors + JANGTQ runtime constants (handled outside the affine path).
        """
        for needle in (
            "_norm.weight", "norm.weight",
            "attn_sink",
            "hc_attn_", "hc_ffn_", "hc_head_",
            "ffn.gate.weight", "ffn.gate.bias",
            "compressor.ape",
            "compressor.norm.weight",
            ".wo_a",
            ".routed_experts.",        # tq_packed (uint32) / tq_norms (bf16)
            "jangtq_",                 # cb_*, signs_*
            "mtp.",                    # MTP block: pre-dequanted bf16 / fp32
        ):
            if needle in key:
                return True
        return False

    def shard(self, group=None):
        return self  # single-device M5 Max; pipeline parallelism deferred

    # -- Sanitize ----------------------------------------------------------

    def sanitize(self, weights: dict) -> dict:
        """Transform on-disk JANGTQ weights into the model's runtime namespace.

        Steps:
          1. Drop `mtp.0.*` keys (MTP head — JANGTQ marks drop_mtp=true).
          2. wo_a: dequantize the 8-bit affine triple (weight, scales, biases)
             and reshape to (n_groups, o_lora_rank, in_per_group). Emit a
             single `layers.{L}.attn.wo_a` tensor and drop the triple.
          3. Routed experts: stack `ffn.experts.{e}.{w1|w2|w3}.tq_{packed,norms}`
             across the 256 experts per layer. Emit
             `layers.{L}.ffn.routed_experts.w{1,2,3}_tq_{packed,norms}` of shape
             (n_experts, out, ...). Drop `tq_bits` (always 2 here — kernel hard-codes).
          4. JANGTQ sidecar (`codebook.{N}.2`, `signs.{N}.42`) — these arrive
             only when `jangtq_runtime.safetensors` is symlinked under a
             `model*.safetensors` name so mlx_lm picks it up. Remap to model
             attribute paths (`jangtq_cb_*`, `jangtq_signs_*`). If absent,
             raise — the model cannot run without the codebook + signs.
          5. tid2eid: pass through (i32 lookup table, hash-routing layers).
          6. Everything else (8-bit affine: attention/shared/indexer/embed/head,
             plus fp16: norms/router/HC/attn_sink, compressor.ape/norm.weight):
             pass through.
        """
        a = self.args
        out: dict[str, mx.array] = {}

        # Bucket wo_a triples by layer for grouped dequant.
        wo_a_buckets: dict[int, dict[str, mx.array]] = defaultdict(dict)
        # Bucket routed-expert tq_* tensors by (layer, sub_weight, kind).
        # routed_buckets[(L, "w1", "tq_packed")][expert_id] = tensor
        routed_buckets: dict[tuple[int, str, str], dict[int, mx.array]] = defaultdict(dict)
        sidecar_seen: dict[str, mx.array] = {}

        for k, v in weights.items():
            # Step 1: MTP. JANGTQ has drop_mtp=true (zero mtp.* keys present),
            # so this branch only fires if a future checkpoint includes them.
            # Always drop here — MTP block weights come from a separate
            # non-JANGTQ load path below, not from the JANGTQ checkpoint.
            if k.startswith("mtp."):
                continue

            # Step 4: JANGTQ runtime sidecar keys (no "layers." prefix).
            if k.startswith("codebook.") or k.startswith("signs."):
                sidecar_seen[k] = v
                continue

            # Step 3: routed-expert quant tensors → stack per layer.
            if ".ffn.experts." in k and (
                ".tq_packed" in k or ".tq_norms" in k or ".tq_bits" in k
            ):
                # Key form: layers.{L}.ffn.experts.{E}.{w1|w2|w3}.{tq_packed|tq_norms|tq_bits}
                parts = k.split(".")
                layer_idx = int(parts[1])
                expert_idx = int(parts[4])
                sub_w = parts[5]                # w1|w2|w3
                kind = parts[6]                 # tq_packed|tq_norms|tq_bits
                if kind == "tq_bits":
                    continue                    # always 2 for routed experts; hard-coded
                routed_buckets[(layer_idx, sub_w, kind)][expert_idx] = v
                continue

            # Step 2: bucket wo_a triple per layer.
            if ".attn.wo_a." in k:
                parts = k.split(".")
                layer_idx = int(parts[1])
                field_name = parts[-1]          # weight|scales|biases
                wo_a_buckets[layer_idx][field_name] = v
                continue

            out[k] = v

        # Dequant wo_a per layer.
        for layer_idx, triple in wo_a_buckets.items():
            w = triple.get("weight")
            s = triple.get("scales")
            b = triple.get("biases")
            if w is None or s is None:
                continue                        # partial — should not happen on a clean checkpoint
            dq = mx.dequantize(w, scales=s, biases=b, group_size=a.group_size, bits=8)
            in_per_group = a.num_attention_heads * a.head_dim // a.o_groups
            dq = dq.reshape(a.o_groups, a.o_lora_rank, in_per_group).astype(mx.bfloat16)
            out[f"layers.{layer_idx}.attn.wo_a"] = dq

        # Stack routed-expert tq_* across experts per layer.
        for (layer_idx, sub_w, kind), per_expert in routed_buckets.items():
            assert len(per_expert) == a.n_routed_experts, (
                f"layer {layer_idx} {sub_w} {kind}: got {len(per_expert)} experts, "
                f"expected {a.n_routed_experts}"
            )
            stacked = mx.stack(
                [per_expert[e] for e in range(a.n_routed_experts)], axis=0
            )
            out[f"layers.{layer_idx}.ffn.routed_experts.{sub_w}_{kind}"] = stacked

        # JANGTQ sidecar → model attribute paths. Required.
        try:
            out["jangtq_cb_4096"] = sidecar_seen["codebook.4096.2"].astype(mx.float32)
            out["jangtq_cb_2048"] = sidecar_seen["codebook.2048.2"].astype(mx.float32)
            out["jangtq_signs_4096"] = sidecar_seen["signs.4096.42"].astype(mx.float32)
            out["jangtq_signs_2048"] = sidecar_seen["signs.2048.42"].astype(mx.float32)
        except KeyError as e:
            raise RuntimeError(
                "JANGTQ runtime sidecar keys not found in checkpoint. "
                "Symlink jangtq_runtime.safetensors as model-jangtq-runtime.safetensors "
                "in the model directory so mlx_lm picks it up via the "
                "`model*.safetensors` glob."
            ) from e

        # MTP-native drafter weights. Loaded from a separate non-JANGTQ Flash
        # directory because JANGTQ drops mtp.* keys (drop_mtp=true). Imported
        # lazily so cold-start cost (~78s pre-dequant of ~12 GB bf16 from FP4
        # routed experts + FP8 attention) is paid only when load_mtp=True.
        if a.load_mtp:
            from _load_mtp_weights import load_non_jangtq_mtp_weights
            mtp_weights = load_non_jangtq_mtp_weights(
                source_dir=a.mtp_source_dir,
                n_routed_experts=a.n_routed_experts,
                verbose=True,
            )
            for k, v in mtp_weights.items():
                out[k] = v

        return out
