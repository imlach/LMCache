# SPDX-License-Identifier: Apache-2.0
"""On-disk persistence for hybrid-state snapshots (restart survival).

Background
----------
PR #3284 captures GDN/Mamba recurrent state in a process-local LRU
(``_HYBRID_STATE_CACHE`` in ``lmcache.integration.vllm.vllm_v1_adapter``)
and PR #1 (``hybrid_state_wire``) ships that LRU across instances over P2P.
Neither survives a process restart: the LRU lives in the engine process,
so a vLLM engine restart leaves the receiver with attention-KV chunk files
on disk but no hybrid state — the hit-gate
(``_get_hybrid_state_loadable_tokens``) reports 0 loadable tokens and the
GDN layers recompute from scratch.

This module persists each hybrid-state snapshot to the same local-disk
directory the ``LocalDiskBackend`` writes attention-KV chunks to, using the
*wire codec's* serialized form (so the on-disk bytes are identical to the
P2P payload — one serialization format, two transports). On a later
process start the integration layer reads the snapshot back into its LRU
on the first lookup that needs it, so the hit-gate passes and the prefix
restores token-exact.

Layer discipline
----------------
This module is model-agnostic: it moves opaque ``bytes`` to and from a
file path. The bytes are produced/consumed by the integration layer's
codec (``_encode_hybrid_state_for_wire`` /
``_decode_and_store_hybrid_state_from_wire``), exactly as for the wire
path. Nothing here imports ``lmcache.integration.*`` or vLLM, mirroring
``hybrid_state_wire.py``.

Key discipline
--------------
The artifact filename is derived from the hybrid-state key
``(num_tokens, token_hash)``, where ``token_hash`` is a blake2b digest over
the prefix token ids (see ``_hybrid_state_key`` in the adapter). That digest
is computed *without* the builtin ``hash()`` and is therefore stable across
processes regardless of ``PYTHONHASHSEED`` — the same prefix always maps to
the same artifact, so a snapshot written by one process is found by the next.
The ``@hybrid`` suffix keeps these files visually distinct from the
``...@<chunkhash>@<dtype>.pt`` attention-KV chunk files in the same dir.

Eviction / orphan safety
------------------------
A hybrid artifact on disk is only ever consulted *after* the attention-KV
lookup reports a positive hit for the same prefix (the gate runs inside
``num_external_hit_tokens > 0``). So a hybrid artifact whose KV chunks have
been evicted cannot, on its own, produce a false hit — it is simply never
read. A corrupt or version-mismatched artifact is removed by the adapter
via ``remove_hybrid_state`` on the failed decode. The ``@hybrid`` namespace
keeps these files matchable so a future warm-start (or an external janitor)
can reconcile them against the KV chunk set.
"""

# Standard
from typing import Optional, Tuple
import os
import tempfile

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

HybridStateKey = Tuple[int, str]

#: Filename suffix that distinguishes a hybrid-state artifact from an
#: attention-KV chunk file (``...@<chunkhash>@<dtype>.pt``).
HYBRID_ARTIFACT_SUFFIX: str = "@hybrid.bin"


def hybrid_state_filename(key: HybridStateKey) -> str:
    """Return the on-disk filename for a hybrid-state key.

    Format: ``hybrid@<num_tokens>@<token_hash>@hybrid.bin``. The leading
    ``hybrid@`` namespace and the trailing suffix make these files trivially
    greppable and impossible to confuse with KV-chunk ``.pt`` files when a
    future warm-start scans the directory.
    """
    num_tokens, token_hash = key
    return f"hybrid@{num_tokens}@{token_hash}{HYBRID_ARTIFACT_SUFFIX}"


def hybrid_state_path(disk_path: str, key: HybridStateKey) -> str:
    """Return the absolute path for a hybrid-state artifact under ``disk_path``."""
    return os.path.join(disk_path, hybrid_state_filename(key))


def is_hybrid_state_file(name: str) -> bool:
    """Return whether a filename is a hybrid-state artifact this module owns."""
    return name.startswith("hybrid@") and name.endswith(HYBRID_ARTIFACT_SUFFIX)


def parse_hybrid_state_filename(name: str) -> Optional[HybridStateKey]:
    """Recover the ``(num_tokens, token_hash)`` key from an artifact filename.

    Returns ``None`` for names that aren't hybrid-state artifacts or that
    don't parse (e.g. a half-written temp file, a future schema). Callers
    treat ``None`` as "not mine, leave it alone".
    """
    if not is_hybrid_state_file(name):
        return None
    stem = name[: -len(HYBRID_ARTIFACT_SUFFIX)]
    parts = stem.split("@")
    # ["hybrid", "<num_tokens>", "<token_hash>"]
    if len(parts) != 3 or parts[0] != "hybrid":
        return None
    try:
        num_tokens = int(parts[1])
    except ValueError:
        return None
    token_hash = parts[2]
    if not token_hash:
        return None
    return num_tokens, token_hash


def save_hybrid_state(
    disk_path: str,
    key: HybridStateKey,
    payload_bytes: bytes,
) -> bool:
    """Persist wire-format hybrid-state bytes to local disk.

    Writes to a temp file in the same directory and atomically renames so a
    concurrent reader (or a restart mid-write) never sees a truncated file.
    Idempotent: re-saving the same key overwrites in place.

    Returns ``True`` on success, ``False`` on any I/O error (logged). A
    ``False`` return is non-fatal — the in-memory LRU still holds the
    snapshot for this process; only restart survival is lost for this key.
    """
    final_path = hybrid_state_path(disk_path, key)
    tmp_fd = -1
    tmp_path = ""
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".hybrid-tmp-", dir=disk_path)
        with os.fdopen(tmp_fd, "wb") as f:
            tmp_fd = -1  # fdopen took ownership
            f.write(payload_bytes)
        os.replace(tmp_path, final_path)
        logger.debug(
            "Persisted hybrid state to disk: %s (%d bytes)",
            final_path,
            len(payload_bytes),
        )
        return True
    except OSError:
        logger.warning(
            "Failed to persist hybrid state to %s; restart survival lost "
            "for this key (in-memory LRU unaffected)",
            final_path,
            exc_info=True,
        )
        if tmp_fd >= 0:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def load_hybrid_state(
    disk_path: str,
    key: HybridStateKey,
) -> Optional[bytes]:
    """Read wire-format hybrid-state bytes back from local disk.

    Returns ``None`` if no artifact exists for ``key`` (the common case —
    most prefixes have no persisted hybrid state) or on any read error
    (logged). A ``None`` return is a benign miss: the caller falls back to
    treating the prefix as attention-only and recomputing GDN layers.
    """
    path = hybrid_state_path(disk_path, key)
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning(
            "Failed to read hybrid state from %s; treating as miss",
            path,
            exc_info=True,
        )
        return None


def remove_hybrid_state(disk_path: str, key: HybridStateKey) -> None:
    """Delete a hybrid-state artifact, if present. Never raises."""
    path = hybrid_state_path(disk_path, key)
    try:
        os.remove(path)
        logger.debug("Removed hybrid state artifact: %s", path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Failed to remove hybrid state %s", path, exc_info=True)
