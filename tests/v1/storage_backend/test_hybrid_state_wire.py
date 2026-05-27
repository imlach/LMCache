# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the hybrid-state cross-instance wire format.

Covers:
- Wire schema roundtrip (HybridStateWirePayload encode/decode via msgspec)
- Version mismatch fail-closed (receiver treats bumped version as miss)
- Codec registration: encode/decode through the public surface
- Codec absent: encode returns None, decode returns False
- Pending-key broker: set/pop semantics + idempotent pop after consume
- Backward-compat: BatchedLookupAndGetMsg / Ret with default-None fields
  decode cleanly when the wire bytes were produced by a pre-extension peer
  (msgspec ignores missing fields, picks up defaults)

These tests stay self-contained — no NIXL, no peer connection, no
LMCacheEngine init. The P2P backend's actual handlers are exercised by
``test_p2p_backend_with_controller.py``; this file just verifies the
new schema + codec broker behave correctly under the contract documented
in ``hybrid_state_wire.py``.
"""

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.storage_backend.hybrid_state_wire import (
    HybridStateTensorRecord,
    HybridStateWirePayload,
    WIRE_VERSION,
    decode_and_store_hybrid_state,
    encode_hybrid_state,
    is_hybrid_state_codec_registered,
    pop_pending_hybrid_state_key,
    register_hybrid_state_codec,
    set_pending_hybrid_state_key,
    unregister_hybrid_state_codec,
)
from lmcache.v1.storage_backend.p2p_backend import (
    BatchedLookupAndGetMsg,
    BatchedLookupAndGetRetMsg,
)


# ───────────────────────────────────────────────────────────────────────────
# Wire schema roundtrip
# ───────────────────────────────────────────────────────────────────────────


def _make_sample_payload() -> HybridStateWirePayload:
    """Produce a small but realistic payload (3 records, 1 GDN group)."""
    return HybridStateWirePayload(
        version=WIRE_VERSION,
        num_tokens=25344,
        token_hash="0123456789abcdef0123456789abcdef",
        tensors=[
            HybridStateTensorRecord(
                group_id=0,
                layer_name="model.layers.0.linear_attn",
                state_index=0,
                shape=[10240, 3],
                dtype="bfloat16",
                data=b"\x00" * (10240 * 3 * 2),  # conv state, ~60 KiB
            ),
            HybridStateTensorRecord(
                group_id=0,
                layer_name="model.layers.0.linear_attn",
                state_index=1,
                shape=[48, 128, 128],
                dtype="bfloat16",
                data=b"\x01" * (48 * 128 * 128 * 2),  # temporal state, 1.5 MiB
            ),
            HybridStateTensorRecord(
                group_id=1,
                layer_name="model.layers.4.linear_attn",
                state_index=0,
                shape=[10240, 3],
                dtype="bfloat16",
                data=b"\x02" * (10240 * 3 * 2),
            ),
        ],
    )


def test_wire_payload_roundtrip():
    """encode → decode should preserve all fields byte-for-byte."""
    original = _make_sample_payload()
    encoded = msgspec.msgpack.encode(original)
    decoded = msgspec.msgpack.decode(encoded, type=HybridStateWirePayload)

    assert decoded.version == original.version
    assert decoded.num_tokens == original.num_tokens
    assert decoded.token_hash == original.token_hash
    assert len(decoded.tensors) == len(original.tensors)

    for orig_rec, dec_rec in zip(original.tensors, decoded.tensors):
        assert dec_rec.group_id == orig_rec.group_id
        assert dec_rec.layer_name == orig_rec.layer_name
        assert dec_rec.state_index == orig_rec.state_index
        assert dec_rec.shape == orig_rec.shape
        assert dec_rec.dtype == orig_rec.dtype
        assert dec_rec.data == orig_rec.data


def test_wire_payload_size_realistic():
    """Sanity check: a Qwen3.5-Next-shaped payload encodes to ~75 MiB on the wire."""
    # 48 GDN layers × (60 KiB conv + 1.5 MiB temporal) ≈ 74.9 MiB raw
    payload = HybridStateWirePayload(
        version=WIRE_VERSION,
        num_tokens=25344,
        token_hash="a" * 32,
        tensors=[
            HybridStateTensorRecord(
                group_id=layer // 16,
                layer_name=f"model.layers.{layer * 4 + 1}.linear_attn",
                state_index=state_idx,
                shape=[10240, 3] if state_idx == 0 else [48, 128, 128],
                dtype="bfloat16",
                data=b"\x00" * (10240 * 3 * 2 if state_idx == 0 else 48 * 128 * 128 * 2),
            )
            for layer in range(48)
            for state_idx in range(2)
        ],
    )
    encoded = msgspec.msgpack.encode(payload)
    # Allow up to 1% overhead for msgpack framing
    raw_bytes = 48 * (10240 * 3 * 2 + 48 * 128 * 128 * 2)
    assert raw_bytes <= len(encoded) <= int(raw_bytes * 1.01) + 4096, (
        f"Encoded size {len(encoded)} outside expected range around {raw_bytes}"
    )


# ───────────────────────────────────────────────────────────────────────────
# Codec registration surface
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_codec():
    """Reset codec + broker state around each test for isolation."""
    unregister_hybrid_state_codec()
    yield
    unregister_hybrid_state_codec()


def test_codec_absent_returns_none():
    """Without a registered codec, encode→None and decode→False (benign miss)."""
    assert not is_hybrid_state_codec_registered()
    assert encode_hybrid_state((100, "abc")) is None
    assert decode_and_store_hybrid_state(b"any bytes") is False


def test_codec_registration_roundtrip():
    """Register → encode/decode dispatch correctly → unregister restores absent state."""
    stored: dict[tuple[int, str], bytes] = {}

    def encoder(key):
        if key not in stored:
            return None
        return msgspec.msgpack.encode(
            HybridStateWirePayload(
                version=WIRE_VERSION,
                num_tokens=key[0],
                token_hash=key[1],
                tensors=[],
            )
        )

    def decoder(payload_bytes):
        wire = msgspec.msgpack.decode(payload_bytes, type=HybridStateWirePayload)
        stored[(wire.num_tokens, wire.token_hash)] = payload_bytes
        return True

    register_hybrid_state_codec(encoder=encoder, decoder=decoder)
    assert is_hybrid_state_codec_registered()

    # Encode against an empty store → None (key not present)
    assert encode_hybrid_state((100, "deadbeef")) is None

    # Seed and re-encode → bytes
    stored[(100, "deadbeef")] = b"seed"
    encoded = encode_hybrid_state((100, "deadbeef"))
    assert encoded is not None

    # Decode → True, populates store under the same key
    stored.clear()
    assert decode_and_store_hybrid_state(encoded) is True
    assert (100, "deadbeef") in stored

    unregister_hybrid_state_codec()
    assert not is_hybrid_state_codec_registered()


def test_codec_exception_is_benign():
    """An encoder/decoder that raises returns None/False, not propagating."""

    def boom_encoder(key):
        raise RuntimeError("synthetic")

    def boom_decoder(payload_bytes):
        raise RuntimeError("synthetic")

    register_hybrid_state_codec(encoder=boom_encoder, decoder=boom_decoder)
    assert encode_hybrid_state((1, "x")) is None
    assert decode_and_store_hybrid_state(b"x") is False


# ───────────────────────────────────────────────────────────────────────────
# Pending-key broker
# ───────────────────────────────────────────────────────────────────────────


def test_pending_key_broker_set_pop():
    """Set then pop returns the key; second pop returns None."""
    set_pending_hybrid_state_key("req-1", (25344, "abc"))
    popped = pop_pending_hybrid_state_key("req-1")
    assert popped == (25344, "abc")
    assert pop_pending_hybrid_state_key("req-1") is None


def test_pending_key_broker_isolation():
    """Different lookup_ids don't interfere."""
    set_pending_hybrid_state_key("req-1", (100, "a"))
    set_pending_hybrid_state_key("req-2", (200, "b"))
    assert pop_pending_hybrid_state_key("req-2") == (200, "b")
    assert pop_pending_hybrid_state_key("req-1") == (100, "a")


def test_pending_key_broker_missing_returns_none():
    """Popping a key that was never set returns None without raising."""
    assert pop_pending_hybrid_state_key("never-set") is None


# ───────────────────────────────────────────────────────────────────────────
# Backward compatibility: P2P message decode with default-None fields
# ───────────────────────────────────────────────────────────────────────────


def test_get_msg_pre_extension_decodes_with_defaults():
    """Old senders emit no ``hybrid_state_key`` / ``msg_version`` fields.

    msgspec drops them on encode (defaults are absent from the wire form
    when the field equals its declared default) and re-fills them on
    decode. A pre-extension wire snippet should decode to defaults None / 0.
    """
    pre_ext_msg = BatchedLookupAndGetMsg(
        lookup_id="x", receiver_id="y", keys=[], mem_indexes=[],
    )
    wire = msgspec.msgpack.encode(pre_ext_msg)
    decoded = msgspec.msgpack.decode(wire, type=BatchedLookupAndGetMsg)

    assert decoded.hybrid_state_key is None
    assert decoded.msg_version == 0


def test_ret_msg_pre_extension_decodes_with_defaults():
    """Likewise for the reply: legacy reply has no hybrid_state_bytes."""
    pre_ext_ret = BatchedLookupAndGetRetMsg(num_hit_chunks=3)
    wire = msgspec.msgpack.encode(pre_ext_ret)
    decoded = msgspec.msgpack.decode(wire, type=BatchedLookupAndGetRetMsg)

    assert decoded.hybrid_state_bytes is None
    assert decoded.num_hit_chunks == 3


def test_get_msg_with_hybrid_extension_roundtrip():
    """New senders include the optional fields; they roundtrip cleanly."""
    new_msg = BatchedLookupAndGetMsg(
        lookup_id="x",
        receiver_id="y",
        keys=["k1", "k2"],
        mem_indexes=[0, 1],
        hybrid_state_key=(25344, "deadbeef"),
        msg_version=1,
    )
    wire = msgspec.msgpack.encode(new_msg)
    decoded = msgspec.msgpack.decode(wire, type=BatchedLookupAndGetMsg)

    assert decoded.hybrid_state_key == (25344, "deadbeef")
    assert decoded.msg_version == 1
    assert decoded.keys == ["k1", "k2"]


def test_ret_msg_with_hybrid_extension_roundtrip():
    """New replies carry hybrid_state_bytes; roundtrip preserves them."""
    payload_bytes = msgspec.msgpack.encode(_make_sample_payload())
    new_ret = BatchedLookupAndGetRetMsg(
        num_hit_chunks=5, hybrid_state_bytes=payload_bytes,
    )
    wire = msgspec.msgpack.encode(new_ret)
    decoded = msgspec.msgpack.decode(wire, type=BatchedLookupAndGetRetMsg)

    assert decoded.num_hit_chunks == 5
    assert decoded.hybrid_state_bytes == payload_bytes


# ───────────────────────────────────────────────────────────────────────────
# Version-mismatch fail-closed
# ───────────────────────────────────────────────────────────────────────────


def test_decoder_rejects_version_mismatch():
    """A payload with version != WIRE_VERSION should be rejected at decode time.

    This is the integration-layer codec's responsibility, not the wire
    module's — the wire module is a transport. We test it here by
    registering a strict decoder that mirrors the integration layer's
    contract, and verifying it rejects a bumped version.
    """
    def strict_decoder(payload_bytes):
        wire = msgspec.msgpack.decode(payload_bytes, type=HybridStateWirePayload)
        return wire.version == WIRE_VERSION

    register_hybrid_state_codec(
        encoder=lambda _: None,  # not exercised in this test
        decoder=strict_decoder,
    )

    bumped = HybridStateWirePayload(
        version=WIRE_VERSION + 1,
        num_tokens=1,
        token_hash="x",
        tensors=[],
    )
    bumped_bytes = msgspec.msgpack.encode(bumped)
    assert decode_and_store_hybrid_state(bumped_bytes) is False
