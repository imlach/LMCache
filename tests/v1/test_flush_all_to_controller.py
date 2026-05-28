# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``LMCacheEngine.flush_all_to_controller``.

Bypasses the full engine __init__ — too many dependencies (GPU connector,
token database, NUMA detection, etc.) for what is fundamentally a tiny piece
of plumbing logic. Constructs the engine via ``__new__`` and pokes the two
attributes the method touches (``storage_manager`` and ``config``) with
lightweight fakes.

What's covered:
- The method iterates every storage backend.
- It calls ``add_kv_op(OpType.ADMIT, key.chunk_hash)`` once per key.
- It calls ``sender.flush()`` once per backend (drain semantics).
- Backends without ``batched_msg_sender`` are skipped (no crash, no admit).
- Backends without ``get_keys`` are skipped (no crash, no admit).
- The returned report shape is correct.
- The order matters: admits are added before flush (so flush actually drains them).
"""

# Standard
from collections import OrderedDict
from types import SimpleNamespace

# First Party
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.cache_engine import LMCacheEngine


class _FakeKey:
    """Mimic CacheEngineKey's only attribute the flush path touches."""

    def __init__(self, chunk_hash: int):
        self.chunk_hash = chunk_hash


class _FakeSender:
    """Records every add_kv_op call + flush invocation order."""

    def __init__(self):
        self.calls: list = []  # interleaved ("admit", chunk_hash) and ("flush",)
        self.flushed_count = 0

    def add_kv_op(self, op_type, key):
        self.calls.append((op_type, key))

    def flush(self):
        # Snapshot the admit count at the moment of flush so we can assert
        # "admits happen *before* flush, not after."
        self.flushed_count = sum(1 for c in self.calls if c[0] == OpType.ADMIT)
        self.calls.append(("flush",))


class _FakeBackend:
    """Configurable backend stub: choose whether it has a sender and/or get_keys."""

    def __init__(self, *, keys=None, sender=None, with_get_keys: bool = True):
        if sender is not None:
            self.batched_msg_sender = sender
        if with_get_keys:
            self._keys = list(keys or [])

    def get_keys(self):
        return self._keys


def _engine_with_backends(backends: "OrderedDict[str, _FakeBackend]"):
    """Construct an LMCacheEngine instance without running __init__."""
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.storage_manager = SimpleNamespace(storage_backends=backends)
    engine.config = SimpleNamespace(lmcache_instance_id="test-instance")
    return engine


def test_admits_every_key_then_flushes():
    sender = _FakeSender()
    backend = _FakeBackend(
        keys=[_FakeKey(101), _FakeKey(202), _FakeKey(303)],
        sender=sender,
    )
    engine = _engine_with_backends(OrderedDict([("LocalCPUBackend", backend)]))

    result = engine.flush_all_to_controller()

    admits = [c for c in sender.calls if c[0] == OpType.ADMIT]
    assert admits == [(OpType.ADMIT, 101), (OpType.ADMIT, 202), (OpType.ADMIT, 303)]
    # flush() must come last + see all 3 admits already queued
    assert sender.calls[-1] == ("flush",)
    assert sender.flushed_count == 3

    assert result == {
        "instance_id": "test-instance",
        "backends": [{"location": "LocalCPUBackend", "keys_admitted": 3}],
        "total_keys_admitted": 3,
    }


def test_iterates_multiple_backends_in_order():
    cpu_sender = _FakeSender()
    disk_sender = _FakeSender()
    backends = OrderedDict(
        [
            (
                "LocalCPUBackend",
                _FakeBackend(keys=[_FakeKey(1), _FakeKey(2)], sender=cpu_sender),
            ),
            (
                "LocalDiskBackend",
                _FakeBackend(keys=[_FakeKey(9)], sender=disk_sender),
            ),
        ]
    )
    engine = _engine_with_backends(backends)

    result = engine.flush_all_to_controller()

    assert cpu_sender.flushed_count == 2
    assert disk_sender.flushed_count == 1
    assert result["total_keys_admitted"] == 3
    assert [b["location"] for b in result["backends"]] == [
        "LocalCPUBackend",
        "LocalDiskBackend",
    ]


def test_skips_backend_without_sender():
    """A backend with no batched_msg_sender is silently skipped."""
    cpu_sender = _FakeSender()
    backends = OrderedDict(
        [
            (
                "LocalCPUBackend",
                _FakeBackend(keys=[_FakeKey(7)], sender=cpu_sender),
            ),
            # PDBackend in the live config has no batched_msg_sender
            ("PDBackend", _FakeBackend(keys=[_FakeKey(8)], sender=None)),
        ]
    )
    engine = _engine_with_backends(backends)

    result = engine.flush_all_to_controller()

    assert cpu_sender.flushed_count == 1
    assert result["total_keys_admitted"] == 1
    assert [b["location"] for b in result["backends"]] == ["LocalCPUBackend"]


def test_skips_backend_without_get_keys():
    """A backend that doesn't expose get_keys is silently skipped."""
    sender = _FakeSender()
    weird_backend = _FakeBackend(sender=sender, with_get_keys=False)
    engine = _engine_with_backends(OrderedDict([("WeirdBackend", weird_backend)]))

    result = engine.flush_all_to_controller()

    assert sender.calls == []  # nothing called
    assert result == {
        "instance_id": "test-instance",
        "backends": [],
        "total_keys_admitted": 0,
    }


def test_empty_backend_still_flushes():
    """A backend with zero keys still gets flush() called — drains stragglers from add path."""
    sender = _FakeSender()
    backend = _FakeBackend(keys=[], sender=sender)
    engine = _engine_with_backends(OrderedDict([("LocalCPUBackend", backend)]))

    result = engine.flush_all_to_controller()

    assert sender.calls == [("flush",)]
    assert result["total_keys_admitted"] == 0
    assert result["backends"] == [{"location": "LocalCPUBackend", "keys_admitted": 0}]


def test_idempotent_when_called_twice():
    """Two back-to-back calls send the admits twice (controller dedupes downstream)."""
    sender = _FakeSender()
    backend = _FakeBackend(keys=[_FakeKey(42)], sender=sender)
    engine = _engine_with_backends(OrderedDict([("LocalCPUBackend", backend)]))

    engine.flush_all_to_controller()
    engine.flush_all_to_controller()

    admits = [c for c in sender.calls if c[0] == OpType.ADMIT]
    assert admits == [(OpType.ADMIT, 42), (OpType.ADMIT, 42)]
