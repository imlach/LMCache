# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import validate_kv_cache_groups


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]
    kv_cache_spec: object


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]


class Qwen3MTPAttentionSpec:
    block_size = 16
    page_size_bytes = 64


class EagleSpec:
    block_size = 16
    page_size_bytes = 64


class FutureDraftCacheSpec:
    block_size = 16


def test_validate_kv_cache_groups_ignores_speculative_attention_specs() -> None:
    validate_kv_cache_groups(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["mtp0"], Qwen3MTPAttentionSpec()),
                MockKVCacheGroup(["eagle0"], EagleSpec()),
            ]
        )
    )


def test_validate_kv_cache_groups_rejects_unknown_spec_classes() -> None:
    with pytest.raises(ValueError, match="FutureDraftCacheSpec"):
        validate_kv_cache_groups(
            MockKVCacheConfig(
                kv_cache_groups=[MockKVCacheGroup(["future0"], FutureDraftCacheSpec())]
            )
        )
