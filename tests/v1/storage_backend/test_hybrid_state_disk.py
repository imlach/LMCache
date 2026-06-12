# SPDX-License-Identifier: Apache-2.0
"""Unit tests for on-disk hybrid-state persistence (restart survival).

Covers the standalone disk module (``hybrid_state_disk``):
- filename derivation + parse roundtrip, suffix discrimination
- save → load roundtrip (bytes preserved exactly)
- atomic overwrite (re-save same key)
- load miss returns None (no artifact, or unreadable)
- remove is idempotent
- a full codec→disk→codec roundtrip: encode a payload through the wire
  codec, persist it, clear the LRU (simulating a process restart), read it
  back from disk, decode into a fresh LRU — proving the on-disk form is the
  same serialization the P2P path uses and that it survives an LRU wipe.

These tests are self-contained: no vLLM, no GPU, no LMCacheEngine. The
codec-roundtrip test registers a tiny in-dict LRU + codec via the public
``hybrid_state_wire`` surface, mirroring how the integration layer wires
its real ``_HYBRID_STATE_CACHE``.
"""

# Standard
import os

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.storage_backend.hybrid_state_disk import (
    HYBRID_ARTIFACT_SUFFIX,
    hybrid_state_filename,
    hybrid_state_path,
    is_hybrid_state_file,
    load_hybrid_state,
    parse_hybrid_state_filename,
    remove_hybrid_state,
    save_hybrid_state,
)
from lmcache.v1.storage_backend.hybrid_state_wire import (
    HybridStateTensorRecord,
    HybridStateWirePayload,
    WIRE_VERSION,
    decode_and_store_hybrid_state,
    encode_hybrid_state,
    register_hybrid_state_codec,
    unregister_hybrid_state_codec,
)


SAMPLE_KEY = (25344, "0123456789abcdef0123456789abcdef")


def _sample_bytes() -> bytes:
    """A small wire payload (2 records) serialized to bytes."""
    payload = HybridStateWirePayload(
        version=WIRE_VERSION,
        num_tokens=SAMPLE_KEY[0],
        token_hash=SAMPLE_KEY[1],
        tensors=[
            HybridStateTensorRecord(
                group_id=0,
                layer_name="model.layers.0.linear_attn",
                state_index=0,
                shape=[10240, 3],
                dtype="bfloat16",
                data=b"\x00" * (10240 * 3 * 2),
            ),
            HybridStateTensorRecord(
                group_id=0,
                layer_name="model.layers.0.linear_attn",
                state_index=1,
                shape=[48, 128, 128],
                dtype="bfloat16",
                data=b"\x01" * (48 * 128 * 128 * 2),
            ),
        ],
    )
    return msgspec.msgpack.encode(payload)


# ───────────────────────────────────────────────────────────────────────────
# Filename derivation
# ───────────────────────────────────────────────────────────────────────────


def test_filename_format_and_suffix():
    name = hybrid_state_filename(SAMPLE_KEY)
    assert name == f"hybrid@25344@{SAMPLE_KEY[1]}{HYBRID_ARTIFACT_SUFFIX}"
    assert is_hybrid_state_file(name)


def test_filename_parse_roundtrip():
    name = hybrid_state_filename(SAMPLE_KEY)
    assert parse_hybrid_state_filename(name) == SAMPLE_KEY


def test_kv_chunk_file_is_not_a_hybrid_file():
    # A representative attention-KV chunk filename written by LocalDiskBackend.
    kv_name = "Qwen3.6-27B@1@0@a1b2c3d4e5f6@uint8.pt"
    assert not is_hybrid_state_file(kv_name)
    assert parse_hybrid_state_filename(kv_name) is None


def test_parse_rejects_malformed():
    assert parse_hybrid_state_filename("hybrid@notanint@hash@hybrid.bin") is None
    assert parse_hybrid_state_filename("hybrid@123@@hybrid.bin") is None
    assert parse_hybrid_state_filename("random.txt") is None


def test_path_joins_under_disk_dir(tmp_path):
    p = hybrid_state_path(str(tmp_path), SAMPLE_KEY)
    assert os.path.dirname(p) == str(tmp_path)
    assert os.path.basename(p) == hybrid_state_filename(SAMPLE_KEY)


# ───────────────────────────────────────────────────────────────────────────
# Save / load / remove
# ───────────────────────────────────────────────────────────────────────────


def test_save_then_load_roundtrip(tmp_path):
    data = _sample_bytes()
    assert save_hybrid_state(str(tmp_path), SAMPLE_KEY, data) is True
    assert os.path.exists(hybrid_state_path(str(tmp_path), SAMPLE_KEY))
    loaded = load_hybrid_state(str(tmp_path), SAMPLE_KEY)
    assert loaded == data


def test_save_overwrites_in_place(tmp_path):
    save_hybrid_state(str(tmp_path), SAMPLE_KEY, b"first")
    save_hybrid_state(str(tmp_path), SAMPLE_KEY, b"second-longer")
    assert load_hybrid_state(str(tmp_path), SAMPLE_KEY) == b"second-longer"
    # No leftover temp files from the atomic-rename dance.
    leftovers = [n for n in os.listdir(tmp_path) if n.startswith(".hybrid-tmp-")]
    assert leftovers == []


def test_load_miss_returns_none(tmp_path):
    assert load_hybrid_state(str(tmp_path), (999, "deadbeef")) is None


def test_remove_is_idempotent(tmp_path):
    save_hybrid_state(str(tmp_path), SAMPLE_KEY, b"x")
    remove_hybrid_state(str(tmp_path), SAMPLE_KEY)
    assert load_hybrid_state(str(tmp_path), SAMPLE_KEY) is None
    # Second remove on an absent file must not raise.
    remove_hybrid_state(str(tmp_path), SAMPLE_KEY)


def test_save_failure_on_bad_dir_returns_false():
    # Non-existent directory → mkstemp raises OSError → save returns False.
    assert save_hybrid_state("/nonexistent/dir/xyz", SAMPLE_KEY, b"x") is False


# ───────────────────────────────────────────────────────────────────────────
# Full codec → disk → codec roundtrip (the restart-survival path)
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def codec_lru():
    """A minimal LRU + codec wired through the public wire surface.

    Mirrors the integration layer: an encoder that reads from a dict-LRU and
    a decoder that writes into it. Reset around each test.
    """
    lru: dict = {}

    def encoder(key):
        payload = lru.get(key)
        if payload is None:
            return None
        # ``payload`` here is already wire bytes for test simplicity.
        return payload

    def decoder(payload_bytes):
        wire = msgspec.msgpack.decode(payload_bytes, type=HybridStateWirePayload)
        if wire.version != WIRE_VERSION:
            return False
        lru[(wire.num_tokens, wire.token_hash)] = payload_bytes
        return True

    register_hybrid_state_codec(encoder=encoder, decoder=decoder)
    yield lru
    unregister_hybrid_state_codec()


def test_persist_survives_lru_wipe(tmp_path, codec_lru):
    """Encode → persist → wipe LRU (restart) → reload from disk → decode."""
    data = _sample_bytes()
    # Process 1: capture into LRU, encode, persist to disk.
    codec_lru[SAMPLE_KEY] = data
    encoded = encode_hybrid_state(SAMPLE_KEY)
    assert encoded is not None
    assert save_hybrid_state(str(tmp_path), SAMPLE_KEY, encoded) is True

    # Simulate engine restart: LRU is empty, only the disk artifact remains.
    codec_lru.clear()
    assert encode_hybrid_state(SAMPLE_KEY) is None  # gate would report a miss

    # Process 2: lookup misses the LRU, falls back to disk, decodes back in.
    from_disk = load_hybrid_state(str(tmp_path), SAMPLE_KEY)
    assert from_disk is not None
    assert decode_and_store_hybrid_state(from_disk) is True
    # The gate would now find it.
    assert encode_hybrid_state(SAMPLE_KEY) is not None
    assert SAMPLE_KEY in codec_lru


def test_corrupt_artifact_decode_fails(tmp_path, codec_lru):
    """A version-bumped (unreadable-by-this-version) artifact decodes False."""
    bumped = HybridStateWirePayload(
        version=WIRE_VERSION + 1,
        num_tokens=SAMPLE_KEY[0],
        token_hash=SAMPLE_KEY[1],
        tensors=[],
    )
    save_hybrid_state(str(tmp_path), SAMPLE_KEY, msgspec.msgpack.encode(bumped))
    from_disk = load_hybrid_state(str(tmp_path), SAMPLE_KEY)
    assert from_disk is not None
    assert decode_and_store_hybrid_state(from_disk) is False
