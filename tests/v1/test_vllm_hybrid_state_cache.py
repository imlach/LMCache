# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
import lmcache.integration.vllm.vllm_v1_adapter as vllm_v1_adapter
from lmcache.integration.vllm.vllm_v1_adapter import (
    HybridStateGroupSpec,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    _get_hybrid_state_payload,
    _hybrid_state_key,
    _hybrid_state_payload_nbytes,
    _put_hybrid_state_payload,
)
from lmcache.v1.storage_backend.hybrid_state_disk import (
    hybrid_state_path,
    is_hybrid_state_file,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def setup_function() -> None:
    with vllm_v1_adapter._HYBRID_STATE_CACHE_LOCK:
        vllm_v1_adapter._HYBRID_STATE_CACHE.clear()
        vllm_v1_adapter._HYBRID_STATE_CACHE_BYTES = 0


def test_hybrid_state_payload_nbytes_counts_tensor_storage() -> None:
    payload = {
        (0, "layer0", 0): torch.zeros(8, dtype=torch.int8),
        (0, "layer1", 0): torch.zeros(4, dtype=torch.float16),
    }

    assert _hybrid_state_payload_nbytes(payload) == 16


def test_hybrid_state_cache_evicts_least_recently_used_by_byte_limit() -> None:
    payload_a = {(0, "layer0", 0): torch.ones(4, dtype=torch.int8)}
    payload_b = {(0, "layer0", 0): torch.ones(4, dtype=torch.int8) * 2}
    payload_c = {(0, "layer0", 0): torch.ones(4, dtype=torch.int8) * 3}

    _put_hybrid_state_payload((4, "a"), payload_a, max_bytes=8)
    _put_hybrid_state_payload((4, "b"), payload_b, max_bytes=8)
    assert _get_hybrid_state_payload((4, "a")) is payload_a

    evicted = _put_hybrid_state_payload((4, "c"), payload_c, max_bytes=8)

    assert evicted == 1
    assert _get_hybrid_state_payload((4, "a")) is payload_a
    assert _get_hybrid_state_payload((4, "b")) is None
    assert _get_hybrid_state_payload((4, "c")) is payload_c


def test_hybrid_state_cache_evicts_by_byte_limit_but_keeps_new_entry() -> None:
    payload_a = {(0, "layer0", 0): torch.ones(8, dtype=torch.int8)}
    payload_b = {(0, "layer0", 0): torch.ones(8, dtype=torch.int8) * 2}
    payload_large = {(0, "layer0", 0): torch.ones(32, dtype=torch.int8)}

    _put_hybrid_state_payload((8, "a"), payload_a, max_bytes=16)
    _put_hybrid_state_payload((8, "b"), payload_b, max_bytes=16)
    evicted = _put_hybrid_state_payload((32, "large"), payload_large, max_bytes=16)

    assert evicted == 2
    assert _get_hybrid_state_payload((8, "a")) is None
    assert _get_hybrid_state_payload((8, "b")) is None
    assert _get_hybrid_state_payload((32, "large")) is payload_large
    assert vllm_v1_adapter._HYBRID_STATE_CACHE_BYTES == 32


def test_hybrid_state_group_selection_keeps_full_attention_for_lmcache() -> None:
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((2, 4),),
        dtypes=(torch.float16,),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["mamba0"], mamba_spec),
            KVCacheGroupSpec(["mamba1"], mamba_spec),
            KVCacheGroupSpec(["attn0", "attn1"], attn_spec),
        ],
    )

    group_id, layer_names, block_size = vllm_v1_adapter._select_lmcache_kv_cache_group(
        kv_cache_config
    )
    hybrid_groups = vllm_v1_adapter._select_hybrid_state_kv_cache_groups(
        kv_cache_config
    )

    assert group_id == 2
    assert layer_names == ("attn0", "attn1")
    assert block_size == 16
    assert [group.group_id for group in hybrid_groups] == [0, 1]
    assert [group.layer_names for group in hybrid_groups] == [
        ("mamba0",),
        ("mamba1",),
    ]


def test_hybrid_state_hit_is_not_loadable_when_state_is_missing() -> None:
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._hybrid_state_kv_cache_groups = (
        HybridStateGroupSpec(0, ("mamba0",), 4, 4),
    )
    connector._hybrid_state_alignment_tokens = 4

    loadable_tokens = connector._get_hybrid_state_loadable_tokens(
        [1, 2, 3, 4],
        num_external_hit_tokens=4,
    )

    assert loadable_tokens == 0


def test_hybrid_state_store_and_load_round_trips_raw_pages() -> None:
    token_ids = [10, 11, 12, 13]
    conv_state_pages = torch.arange(16, dtype=torch.int8).reshape(4, 4)[1:]
    ssm_state_pages = (torch.arange(8, dtype=torch.int8) + 20).reshape(4, 2)[1:]
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._hybrid_state_kv_cache_groups = (
        HybridStateGroupSpec(0, ("mamba0",), 4, 6),
    )
    connector._hybrid_state_alignment_tokens = 4
    connector._hybrid_state_cache_max_bytes = 1024
    connector._hybrid_state_disk_path = None
    connector._all_kv_caches = {"mamba0": [conv_state_pages, ssm_state_pages]}
    request = ReqMeta(
        req_id="req-1",
        token_ids=token_ids,
        slot_mapping=torch.arange(4, dtype=torch.long),
        all_block_ids=([1, 2],),
    )

    connector._store_hybrid_state(request)
    payload = _get_hybrid_state_payload(_hybrid_state_key(token_ids, 4))
    assert payload is not None
    assert set(payload) == {(0, "mamba0", 0), (0, "mamba0", 1)}

    conv_state_pages[1].fill_(0)
    ssm_state_pages[1].fill_(0)
    request.load_spec = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=4,
        can_load=True,
    )

    assert connector._load_hybrid_state(request)
    assert torch.equal(
        conv_state_pages[1], torch.tensor([8, 9, 10, 11], dtype=torch.int8)
    )
    assert torch.equal(ssm_state_pages[1], torch.tensor([24, 25], dtype=torch.int8))


def _make_capture_connector(disk_path, max_bytes=1024):
    """A bare connector wired for hybrid-state capture (one mamba group)."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._hybrid_state_kv_cache_groups = (
        HybridStateGroupSpec(0, ("mamba0",), 4, 6),
    )
    connector._hybrid_state_alignment_tokens = 4
    connector._hybrid_state_cache_max_bytes = max_bytes
    connector._hybrid_state_disk_path = disk_path
    return connector


def test_store_hybrid_state_persists_artifact_to_disk(tmp_path) -> None:
    """Capture on the normal store path must write a disk artifact keyed by
    the blake2b hybrid key — the restart-survival producer in plain local
    (non-P2P) operation."""
    token_ids = [10, 11, 12, 13]
    conv_state_pages = torch.arange(16, dtype=torch.int8).reshape(4, 4)[1:]
    ssm_state_pages = (torch.arange(8, dtype=torch.int8) + 20).reshape(4, 2)[1:]
    connector = _make_capture_connector(str(tmp_path))
    connector._all_kv_caches = {"mamba0": [conv_state_pages, ssm_state_pages]}
    request = ReqMeta(
        req_id="req-disk",
        token_ids=token_ids,
        slot_mapping=torch.arange(4, dtype=torch.long),
        all_block_ids=([1, 2],),
    )

    connector._store_hybrid_state(request)

    key = _hybrid_state_key(token_ids, 4)
    # In-memory LRU populated...
    assert _get_hybrid_state_payload(key) is not None
    # ...and a single greppable @hybrid artifact landed on disk.
    artifacts = [p.name for p in tmp_path.iterdir() if is_hybrid_state_file(p.name)]
    assert artifacts == [hybrid_state_path(str(tmp_path), key).split("/")[-1]]


def test_disk_artifact_repopulates_lru_after_restart(tmp_path) -> None:
    """After a process restart the LRU is empty; the on-disk artifact written
    by a prior process must repopulate it on the first lookup (the smoking-gun
    path: the hit-gate reported 0 matching hybrid state before this worked)."""
    token_ids = [10, 11, 12, 13]
    conv_state_pages = torch.arange(16, dtype=torch.int8).reshape(4, 4)[1:]
    ssm_state_pages = (torch.arange(8, dtype=torch.int8) + 20).reshape(4, 2)[1:]
    connector = _make_capture_connector(str(tmp_path))
    connector._all_kv_caches = {"mamba0": [conv_state_pages, ssm_state_pages]}
    request = ReqMeta(
        req_id="req-restart",
        token_ids=token_ids,
        slot_mapping=torch.arange(4, dtype=torch.long),
        all_block_ids=([1, 2],),
    )
    connector._store_hybrid_state(request)

    key = _hybrid_state_key(token_ids, 4)

    # Simulate the engine-process restart: the in-memory LRU is wiped, but the
    # disk artifact survives.
    with vllm_v1_adapter._HYBRID_STATE_CACHE_LOCK:
        vllm_v1_adapter._HYBRID_STATE_CACHE.clear()
        vllm_v1_adapter._HYBRID_STATE_CACHE_BYTES = 0
    assert _get_hybrid_state_payload(key) is None

    # The gate's per-key reload pulls it back in from disk, token-exact.
    assert connector._ensure_hybrid_state_loaded(key)
    payload = _get_hybrid_state_payload(key)
    assert payload is not None
    assert set(payload) == {(0, "mamba0", 0), (0, "mamba0", 1)}


def test_store_hybrid_state_no_disk_path_keeps_memory_only(tmp_path) -> None:
    """With disk persistence off, capture still populates the in-memory LRU
    and writes nothing to disk (best-effort persistence is decoupled from
    capture)."""
    token_ids = [10, 11, 12, 13]
    conv_state_pages = torch.arange(16, dtype=torch.int8).reshape(4, 4)[1:]
    ssm_state_pages = (torch.arange(8, dtype=torch.int8) + 20).reshape(4, 2)[1:]
    connector = _make_capture_connector(disk_path=None)
    connector._all_kv_caches = {"mamba0": [conv_state_pages, ssm_state_pages]}
    request = ReqMeta(
        req_id="req-nodisk",
        token_ids=token_ids,
        slot_mapping=torch.arange(4, dtype=torch.long),
        all_block_ids=([1, 2],),
    )

    connector._store_hybrid_state(request)

    assert _get_hybrid_state_payload(_hybrid_state_key(token_ids, 4)) is not None
    assert not any(is_hybrid_state_file(p.name) for p in tmp_path.iterdir())


def test_store_hybrid_state_skips_when_block_id_missing(tmp_path) -> None:
    """A prefix with no recurrent-state block (all_block_ids=None) must skip
    capture without raising and without writing an artifact."""
    connector = _make_capture_connector(str(tmp_path))
    connector._all_kv_caches = {"mamba0": [torch.zeros(4, 4, dtype=torch.int8)]}
    request = ReqMeta(
        req_id="req-noblock",
        token_ids=[10, 11, 12, 13],
        slot_mapping=torch.arange(4, dtype=torch.long),
        all_block_ids=None,
    )

    connector._store_hybrid_state(request)

    assert _get_hybrid_state_payload(_hybrid_state_key([10, 11, 12, 13], 4)) is None
    assert not any(is_hybrid_state_file(p.name) for p in tmp_path.iterdir())
