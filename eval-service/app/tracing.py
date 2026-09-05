"""
Thin wrapper around the Langfuse SDK.

Design choice: rather than making every one of the other six services pull
in the Langfuse SDK and hold credentials, this service centralizes
observability. Other services POST step events to us
(`/trace/start`, `/trace/step`, `/trace/end`) and we forward them to
Langfuse. This also means the eval-service can degrade gracefully (log
locally, no-op) if LANGFUSE_* env vars aren't set yet — useful right now,
since agent-service/orchestrator-api aren't finished and this shouldn't
block development.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger("eval-service.tracing")

_LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
_LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
_LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

_client = None
_enabled = False

if _LANGFUSE_PUBLIC_KEY and _LANGFUSE_SECRET_KEY:
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=_LANGFUSE_PUBLIC_KEY,
            secret_key=_LANGFUSE_SECRET_KEY,
            host=_LANGFUSE_HOST,
        )
        _enabled = True
        logger.info("Langfuse tracing enabled (host=%s).", _LANGFUSE_HOST)
    except Exception as exc:  # pragma: no cover - depends on optional dep / network
        logger.warning("Langfuse SDK unavailable (%s); tracing will be logged locally only.", exc)
else:
    logger.warning(
        "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set; tracing will be logged locally only."
    )

# In-memory fallback store so /trace/* endpoints still work end-to-end
# (and are testable) even with Langfuse disabled.
_local_traces: Dict[str, Dict[str, Any]] = {}


def start_trace(name: str, metadata: Optional[dict] = None) -> str:
    trace_id = str(uuid.uuid4())
    _local_traces[trace_id] = {"name": name, "metadata": metadata or {}, "steps": []}
    if _enabled:
        _client.trace(id=trace_id, name=name, metadata=metadata or {})
    else:
        logger.info("[trace:start] id=%s name=%s metadata=%s", trace_id, name, metadata)
    return trace_id


def log_step(
    trace_id: str,
    name: str,
    input: Any = None,
    output: Any = None,
    latency_ms: Optional[float] = None,
    tokens: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> None:
    step = {
        "name": name,
        "input": input,
        "output": output,
        "latency_ms": latency_ms,
        "tokens": tokens,
        "metadata": metadata or {},
    }
    _local_traces.setdefault(trace_id, {"name": "unknown", "metadata": {}, "steps": []})
    _local_traces[trace_id]["steps"].append(step)

    if _enabled:
        span = _client.span(
            trace_id=trace_id,
            name=name,
            input=input,
            output=output,
            metadata={**(metadata or {}), "latency_ms": latency_ms, "tokens": tokens},
        )
        span.end()
    else:
        logger.info("[trace:step] trace_id=%s name=%s latency_ms=%s", trace_id, name, latency_ms)


def end_trace(trace_id: str, output: Any = None) -> None:
    if trace_id in _local_traces:
        _local_traces[trace_id]["output"] = output
    if _enabled:
        _client.trace(id=trace_id, output=output)
        _client.flush()
    else:
        logger.info("[trace:end] trace_id=%s output=%s", trace_id, output)


def get_trace(trace_id: str) -> Optional[dict]:
    return _local_traces.get(trace_id)


def is_langfuse_enabled() -> bool:
    return _enabled
