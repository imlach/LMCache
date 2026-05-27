# SPDX-License-Identifier: Apache-2.0
"""Wire format and codec registration for hybrid-state cross-instance transfer.

Background
----------
PR #3284 added a process-local LRU (``_HYBRID_STATE_CACHE`` in
``lmcache.integration.vllm.vllm_v1_adapter``) that captures GDN/Mamba
recurrent state for prefix re-use within a single instance. The LRU is
*never* shared across instances, so an external prefix hit on a hybrid
model (e.g. Qwen3-Next, Qwen3.5-Next) restores attention KV correctly via
P2P + NIXL but leaves the receiver's GDN layers with empty recurrent state
— producing silently-wrong outputs.

This module defines the wire format that carries hybrid-state payloads
alongside the existing ``BatchedLookupAndGetRetMsg`` reply on the P2P
backend, plus a codec-registration surface so the storage backend stays
model-agnostic.

Layer discipline
----------------
The wire format (``HybridStateWirePayload``) lives here so both the
storage backend (which sends/receives) and the integration adapter
(which owns the LRU) can refer to the same msgspec schema. The actual
encoding/decoding is provided by the integration layer via
``register_hybrid_state_codec`` — keeps ``lmcache.v1.storage_backend.*``
free of imports from ``lmcache.integration.*``.

Backward compatibility
----------------------
- The format carries an explicit ``version`` field; receivers fail closed
  (treat as miss, fall back to recompute) on version mismatch — no risk
  of silent state corruption.
- The codec registration is optional. If no codec is registered (pure-
  attention models, older integration layer), ``encode_hybrid_state`` /
  ``decode_and_store_hybrid_state`` return ``None`` / ``False``, and the
  P2P backend behaves exactly as before this extension.
"""

# Standard
from typing import Callable, Optional, Tuple
import threading

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Wire schema
# ---------------------------------------------------------------------------

#: Wire format version. Bump for breaking changes; receivers fail closed on
#: mismatch (treat as miss → recompute).
WIRE_VERSION: int = 1


class HybridStateTensorRecord(msgspec.Struct):
    """One opaque hybrid-state tensor page slot on the wire.

    Each record carries enough information to reconstruct the original
    ``torch.Tensor`` on the receiver:

    - ``group_id``, ``layer_name``, ``state_index``: identify the slot
      within the receiver's hybrid-state payload dict
      (``HybridStatePayload = dict[tuple[int, str, int], torch.Tensor]``).
    - ``shape`` + ``dtype``: enable a typed view over ``data``.
    - ``data``: raw bytes of the tensor's underlying storage (bfloat16,
      float16, int8, etc. — the integration layer chooses the dtype).
    """

    group_id: int
    layer_name: str
    state_index: int
    shape: list[int]
    dtype: str
    data: bytes


class HybridStateWirePayload(msgspec.Struct):
    """One hybrid-state checkpoint, ready to ship over ZMQ.

    Carries:
    - ``version``: schema version; receivers compare against ``WIRE_VERSION``
    - ``num_tokens`` + ``token_hash``: the ``HybridStateKey`` the receiver
      uses to insert into its local LRU (``(num_tokens, blake2b16_hex)``)
    - ``tensors``: one record per (group, layer, state_index) slot

    Total wire size for Qwen3.5-Next (48 GDN layers × 1.59 MiB) ≈ 75 MiB
    per snapshot. At ~1.5 GB/s sustained over RoCE TCP the round-trip is
    ~100 ms — small relative to attention KV transfer (~170 ms for 25K
    tokens at FP8) and to full re-prefill (~2-3 s on A6000).
    """

    version: int
    num_tokens: int
    token_hash: str
    tensors: list[HybridStateTensorRecord]


# ---------------------------------------------------------------------------
# Codec registration surface
# ---------------------------------------------------------------------------

#: Type alias: encoder takes a hybrid-state key (num_tokens, hash) and
#: returns wire-format bytes, or ``None`` if the key is not in the
#: integration layer's local cache (codec implementation responsibility).
HybridStateEncoder = Callable[[Tuple[int, str]], Optional[bytes]]

#: Type alias: decoder takes wire-format bytes and stores the decoded
#: payload in the integration layer's local cache. Returns ``True`` on
#: success, ``False`` on any decoding error (caller treats as miss).
HybridStateDecoder = Callable[[bytes], bool]


_codec_lock = threading.Lock()
_encoder: Optional[HybridStateEncoder] = None
_decoder: Optional[HybridStateDecoder] = None

# Process-local broker that bridges the integration layer (where the
# scheduler computes the hybrid_state_key) and the storage backend layer
# (where the P2P lookup fires asynchronously, potentially on a different
# thread/event loop). Keyed by lookup_id, which is unique per request and
# already flows through both layers.
#
# Why a broker and not transfer_spec: the GET path that reaches the P2P
# backend is constructed deep inside ``storage_manager.async_lookup_and_prefetch``,
# and threading a new field through engine.lookup → engine.async_lookup_and_prefetch
# → storage_manager.async_lookup_and_prefetch → backend.batched_get_non_blocking
# would touch four API surfaces. A lookup-id-keyed broker is one
# tiny indirection in this module and avoids the API churn.
_pending_keys_lock = threading.Lock()
_pending_keys: dict[str, Tuple[int, str]] = {}


def register_hybrid_state_codec(
    encoder: HybridStateEncoder,
    decoder: HybridStateDecoder,
) -> None:
    """Register a hybrid-state codec for cross-instance wire transfer.

    Called once by the integration layer at engine init. Idempotent:
    re-registration replaces the previous codec (useful for tests).

    The codec is process-global because the hybrid-state LRU it accesses
    is also process-global — a single LMCacheEngine per process owns
    both.
    """
    global _encoder, _decoder
    with _codec_lock:
        if _encoder is not None or _decoder is not None:
            logger.debug(
                "Hybrid-state codec already registered; replacing. "
                "(Expected during test resets; unexpected in production.)"
            )
        _encoder = encoder
        _decoder = decoder
        logger.info("Hybrid-state codec registered for P2P wire transfer")


def unregister_hybrid_state_codec() -> None:
    """Clear the registered codec. Primarily for test teardown."""
    global _encoder, _decoder
    with _codec_lock:
        _encoder = None
        _decoder = None


def is_hybrid_state_codec_registered() -> bool:
    """Return True if a codec is registered (i.e. wire transfer is wired up)."""
    return _encoder is not None and _decoder is not None


def encode_hybrid_state(key: Tuple[int, str]) -> Optional[bytes]:
    """Encode the hybrid-state payload for ``key`` to wire-format bytes.

    Returns ``None`` if:

    - No codec is registered (pure-attention setup or integration layer
      doesn't speak this wire version)
    - The integration layer's local LRU has no entry for ``key``
    - The codec raised an exception (logged at WARNING)

    A ``None`` return is a normal miss — the receiver falls back to
    treating the cross-instance hit as attention-only.
    """
    encoder = _encoder
    if encoder is None:
        return None
    try:
        return encoder(key)
    except Exception:
        logger.warning(
            "Hybrid-state encoder raised; treating as miss (key=%r)",
            key,
            exc_info=True,
        )
        return None


def set_pending_hybrid_state_key(
    lookup_id: str,
    key: Tuple[int, str],
) -> None:
    """Stash the hybrid_state_key the scheduler computed for this lookup_id.

    Called by the integration layer immediately before ``lookup_client.lookup``.
    The P2P backend's ``batched_get_non_blocking`` picks it up via
    ``pop_pending_hybrid_state_key`` when the prefetch fires (potentially
    asynchronously on the storage manager's event loop).

    Keys are scoped per-lookup_id; the broker is process-global because both
    the integration layer and the P2P backend run in the same process.
    """
    with _pending_keys_lock:
        _pending_keys[lookup_id] = key


def pop_pending_hybrid_state_key(
    lookup_id: str,
) -> Optional[Tuple[int, str]]:
    """Consume the pending hybrid_state_key for ``lookup_id``, if any.

    Called by the P2P backend at the moment it constructs ``BatchedLookupAndGetMsg``.
    Returns ``None`` for pure-attention models / pre-extension scheduler paths
    where the integration layer never stashed a key.
    """
    with _pending_keys_lock:
        return _pending_keys.pop(lookup_id, None)


def clear_pending_hybrid_state_key(lookup_id: str) -> None:
    """Drop a stashed key without consuming it (e.g. on lookup abort)."""
    with _pending_keys_lock:
        _pending_keys.pop(lookup_id, None)


def decode_and_store_hybrid_state(payload_bytes: bytes) -> bool:
    """Decode wire-format bytes and store the payload in the integration LRU.

    Returns ``True`` on successful decode + store. Returns ``False`` if:

    - No codec is registered
    - Wire-format version mismatch
    - msgpack decode failure
    - Tensor reconstruction failed (unknown dtype, etc.)

    A ``False`` return is benign: the receiver's scheduler treats the
    cross-instance hit as if hybrid state were unavailable, falls back
    to recompute on the GDN layers.
    """
    decoder = _decoder
    if decoder is None:
        return False
    try:
        return decoder(payload_bytes)
    except Exception:
        logger.warning(
            "Hybrid-state decoder raised; treating as miss",
            exc_info=True,
        )
        return False
