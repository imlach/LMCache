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
    _normalize_hybrid_state_group_block_sizes,
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


def test_hybrid_state_group_selection_ignores_speculative_attention_specs() -> None:
    class Qwen3MTPAttentionSpec:
        block_size = 16
        page_size_bytes = 64

    class EagleSpec:
        block_size = 16
        page_size_bytes = 64

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
            KVCacheGroupSpec(["mtp0"], Qwen3MTPAttentionSpec()),
            KVCacheGroupSpec(["eagle0"], EagleSpec()),
        ],
    )

    hybrid_groups = vllm_v1_adapter._select_hybrid_state_kv_cache_groups(
        kv_cache_config
    )

    assert [group.group_id for group in hybrid_groups] == [0]
    assert [group.layer_names for group in hybrid_groups] == [("mamba0",)]


def test_unknown_kv_cache_specs_are_not_treated_as_hybrid_state(caplog) -> None:
    class FutureDraftCacheSpec:
        pass

    with caplog.at_level("WARNING"):
        is_hybrid = vllm_v1_adapter._is_hybrid_state_kv_cache_spec(
            FutureDraftCacheSpec()
        )

    assert not is_hybrid
    assert "Unknown vLLM KV cache spec" in caplog.text


def test_hybrid_state_block_size_normalization_uses_attention_block_size() -> None:
    hybrid_groups = (
        HybridStateGroupSpec(0, ("mamba0",), 81920, 64),
        HybridStateGroupSpec(1, ("mamba1",), 81920, 64),
    )

    normalized = _normalize_hybrid_state_group_block_sizes(hybrid_groups, 1584)

    assert [group.block_size for group in normalized] == [1584, 1584]
    assert [group.group_id for group in normalized] == [0, 1]
    assert [group.layer_names for group in normalized] == [("mamba0",), ("mamba1",)]


def test_hybrid_state_block_size_normalization_keeps_smaller_block_size() -> None:
    hybrid_groups = (HybridStateGroupSpec(0, ("mamba0",), 1568, 64),)

    assert (
        _normalize_hybrid_state_group_block_sizes(hybrid_groups, 1584) == hybrid_groups
    )


def _make_block_id_connector() -> LMCacheConnectorV1Impl:
    """A bare connector with one mamba group, block_size 1568 (the on-cluster
    Qwen3.6-27B GDN shape)."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._hybrid_state_kv_cache_groups = (
        HybridStateGroupSpec(0, ("mamba0",), 1568, 6),
    )
    connector._hybrid_state_alignment_tokens = 1568
    return connector


def test_get_hybrid_state_block_id_picks_last_non_null_for_padded_mamba_list() -> None:
    """Mirror the real vLLM v0.22.1 mamba block-table shape: ``req_to_blocks``
    for a mamba group (mamba_cache_mode "none"/"align") is a null-padded list
    with the single live recurrent-state page at the END — null pages all carry
    block_id 0 (block_pool.py:173-177). For a 16,611-token request at the 12544
    aligned boundary (8 * 1568), the live state is block 42, NOT the
    token-derived index 7 (which lands on a null block). Regression guard for
    the round-2 capture failure.
    """
    connector = _make_block_id_connector()
    # 8 entries: indices 0..6 are null padding (id 0), the live state is at the
    # last index. This is what get_blocks() returns for a mamba group at this
    # prefix (single_type_kv_cache_manager.py:1040-1084, get_num_skipped_tokens
    # at :1092-1098).
    mamba_block_ids = [0, 0, 0, 0, 0, 0, 0, 42]
    request = ReqMeta(
        req_id="req-padded",
        token_ids=list(range(16611)),
        slot_mapping=torch.arange(1, dtype=torch.long),
        all_block_ids=(mamba_block_ids,),
    )

    group = connector._hybrid_state_kv_cache_groups[0]
    assert connector._get_hybrid_state_block_id(request, group, 12544) == 42


def test_get_hybrid_state_block_id_single_live_page_none_mode() -> None:
    """mamba_cache_mode "none" (default): one physical page per request, so the
    block list is mostly nulls with a single real id. The resolver returns that
    id regardless of how far back it sits."""
    connector = _make_block_id_connector()
    request = ReqMeta(
        req_id="req-none-mode",
        token_ids=list(range(3 * 1568)),
        slot_mapping=torch.arange(1, dtype=torch.long),
        all_block_ids=([0, 0, 17],),
    )
    group = connector._hybrid_state_kv_cache_groups[0]
    assert connector._get_hybrid_state_block_id(request, group, 3 * 1568) == 17


def test_get_hybrid_state_block_id_none_when_all_blocks_null() -> None:
    """An all-null block list (no live recurrent-state page allocated) yields
    None so capture skips with the anomaly WARN rather than reading a null
    page."""
    connector = _make_block_id_connector()
    request = ReqMeta(
        req_id="req-all-null",
        token_ids=list(range(2 * 1568)),
        slot_mapping=torch.arange(1, dtype=torch.long),
        all_block_ids=([0, 0],),
    )
    group = connector._hybrid_state_kv_cache_groups[0]
    assert connector._get_hybrid_state_block_id(request, group, 2 * 1568) is None


def test_get_hybrid_state_block_id_none_for_unaligned_or_missing() -> None:
    """Guards: unaligned token count, missing block ids, and out-of-range group
    id all return None."""
    connector = _make_block_id_connector()
    group = connector._hybrid_state_kv_cache_groups[0]

    unaligned = ReqMeta(
        req_id="req-unaligned",
        token_ids=list(range(1600)),
        slot_mapping=torch.arange(1, dtype=torch.long),
        all_block_ids=([0, 9],),
    )
    # 1600 is not a multiple of block_size 1568.
    assert connector._get_hybrid_state_block_id(request=unaligned, group=group,
                                                num_tokens=1600) is None

    no_blocks = ReqMeta(
        req_id="req-no-blocks",
        token_ids=list(range(1568)),
        slot_mapping=torch.arange(1, dtype=torch.long),
        all_block_ids=None,
    )
    assert connector._get_hybrid_state_block_id(no_blocks, group, 1568) is None

    # Group id out of range (only one block-id group present, group_id is 0,
    # but synthesize a spec whose group_id exceeds the list).
    oor_group = HybridStateGroupSpec(5, ("mamba5",), 1568, 6)
    in_range = ReqMeta(
        req_id="req-oor",
        token_ids=list(range(1568)),
        slot_mapping=torch.arange(1, dtype=torch.long),
        all_block_ids=([0, 3],),
    )
    assert connector._get_hybrid_state_block_id(in_range, oor_group, 1568) is None


def test_store_and_load_round_trips_padded_mamba_block_list(tmp_path) -> None:
    """End-to-end capture+restore using the real null-padded mamba block-table
    shape: the live state page sits at the last (non-null) block id, and capture
    must read/write THAT page — not a token-derived index. Page tensors are
    sized so the live block id indexes a valid row."""
    token_ids = list(range(2 * 1568))
    # 4 physical pages; the live mamba state for this request is page id 3.
    conv_state_pages = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    ssm_state_pages = (torch.arange(8, dtype=torch.int8) + 20).reshape(4, 2)
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._hybrid_state_kv_cache_groups = (
        HybridStateGroupSpec(0, ("mamba0",), 1568, 6),
    )
    connector._hybrid_state_alignment_tokens = 1568
    connector._hybrid_state_cache_max_bytes = 1024
    connector._hybrid_state_disk_path = None
    connector._all_kv_caches = {"mamba0": [conv_state_pages, ssm_state_pages]}
    # Null-padded list: only the last entry (id 3) is the live state page.
    request = ReqMeta(
        req_id="req-padded-rt",
        token_ids=token_ids,
        slot_mapping=torch.arange(1, dtype=torch.long),
        all_block_ids=([0, 3],),
    )

    connector._store_hybrid_state(request)
    payload = _get_hybrid_state_payload(_hybrid_state_key(token_ids, 2 * 1568))
    assert payload is not None
    # Captured the live page (id 3), not a null/token-derived page.
    assert torch.equal(
        payload[(0, "mamba0", 0)].view(torch.int8),
        torch.tensor([12, 13, 14, 15], dtype=torch.int8),
    )

    # Corrupt the live pages, then restore from the captured payload.
    conv_state_pages[3].fill_(0)
    ssm_state_pages[3].fill_(0)
    request.load_spec = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=2 * 1568,
        can_load=True,
    )
    assert connector._load_hybrid_state(request)
    assert torch.equal(
        conv_state_pages[3], torch.tensor([12, 13, 14, 15], dtype=torch.int8)
    )
    assert torch.equal(ssm_state_pages[3], torch.tensor([26, 27], dtype=torch.int8))


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

    # The resolver reads the LAST non-null block (id 2), the live recurrent-state
    # page — not index 0. Page-tensor row 2 holds the live state.
    conv_state_pages[2].fill_(0)
    ssm_state_pages[2].fill_(0)
    request.load_spec = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=4,
        can_load=True,
    )

    assert connector._load_hybrid_state(request)
    assert torch.equal(
        conv_state_pages[2], torch.tensor([12, 13, 14, 15], dtype=torch.int8)
    )
    assert torch.equal(ssm_state_pages[2], torch.tensor([26, 27], dtype=torch.int8))


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
