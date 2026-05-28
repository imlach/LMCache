# SPDX-License-Identifier: Apache-2.0
# Standard
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _get_engine(request: Request):
    adapter = getattr(request.app.state, "lmcache_adapter", None)
    engine = getattr(adapter, "lmcache_engine", None) if adapter else None
    if not engine:
        error_info = {
            "error": "flush_all API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return None, PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=503,
        )
    return engine, None


@router.post("/controller/flush_all")
async def flush_all(request: Request):
    """Drain pending KV-op admits and re-publish every cached key to the controller.

    Wraps :meth:`LMCacheEngine.flush_all_to_controller`. Intended to be
    poked by an orchestrator (e.g. an agentic-review runner at the T0→T1
    tier-switch moment) on the worker instance whose KV cache is about to
    be consulted cross-instance, so the controller's registry reflects
    the full set of locally-held chunk-hashes before the peer looks them
    up.

    Idempotent — safe to call repeatedly.

    Example:
        ```bash
        curl -X POST "http://vllm-review.inference.svc.cluster.local:<port>/controller/flush_all"
        # Response: {
        #   "status": "success",
        #   "instance_id": "vllm-review",
        #   "backends": [{"location": "LocalCPUBackend", "keys_admitted": 117}],
        #   "total_keys_admitted": 117
        # }
        ```
    """
    try:
        engine, err = _get_engine(request)
        if err:
            return err

        result = engine.flush_all_to_controller()  # type: ignore[union-attr]

        return PlainTextResponse(
            content=json.dumps(
                {"status": "success", **result},
                indent=2,
            ),
            media_type="application/json",
        )
    except Exception as e:
        logger.error("flush_all failed: %s", str(e))
        return PlainTextResponse(
            content=json.dumps(
                {"error": "Failed to flush", "message": str(e)},
                indent=2,
            ),
            media_type="application/json",
            status_code=500,
        )
