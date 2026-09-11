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

SDK version note: Langfuse's Python SDK moved to an OpenTelemetry-based
client (v3+) with a completely different API from the older `.trace()` /
`.span()` methods (removed in newer releases). This module is written
against the actual API surface of the installed SDK (`start_observation`,
`create_trace_id`, `get_trace_url`, span `.update()` / `.end()`), verified
by introspecting the installed package rather than assumed from memory or
documentation that may describe an older major version. If the installed
SDK is upgraded again and its API changes, `_enabled` will still correctly
fall back to local-only logging on any import/attribute error, but the
Langfuse-backed calls in `start_trace` / `log_step` / `end_trace` will need
re-checking against whatever the new API looks like.
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

# In-memory fallback store so /trace/* endpoints work end-to-end and are
# testable regardless of whether Langfuse is enabled. This also backs
# `get_trace`, which is used for local inspection even when Langfuse is on.
_local_traces: Dict[str, Dict[str, Any]] = {}

# Holds the live Langfuse root-span object for each open trace_id, so
# log_step/end_trace can attach child observations / close it out. Only
# populated when Langfuse is enabled.
_root_spans: Dict[str, Any] = {}


def start_trace(name: str, metadata: Optional[dict] = None) -> str:
    if _enabled:
        trace_id = _client.create_trace_id()
    else:
        trace_id = str(uuid.uuid4())

    _local_traces[trace_id] = {"name": name, "metadata": metadata or {}, "steps": []}

    if _enabled:
        try:
            root_span = _client.start_observation(
                trace_context={"trace_id": trace_id},
                name=name,
                as_type="span",
                metadata=metadata or {},
            )
            _root_spans[trace_id] = root_span
        except Exception as exc:  # pragma: no cover - network/API-shape failure
            logger.warning("Langfuse start_observation failed (%s); trace %s is local-only.", exc, trace_id)
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

    root_span = _root_spans.get(trace_id)
    if _enabled and root_span is not None:
        try:
            child = root_span.start_observation(
                name=name,
                input=input,
                output=output,
                metadata={**(metadata or {}), "latency_ms": latency_ms, "tokens": tokens},
            )
            child.end()
        except Exception as exc:  # pragma: no cover - network/API-shape failure
            logger.warning("Langfuse step logging failed for trace %s (%s).", trace_id, exc)
    elif not _enabled:
        logger.info("[trace:step] trace_id=%s name=%s latency_ms=%s", trace_id, name, latency_ms)


def end_trace(trace_id: str, output: Any = None) -> None:
    if trace_id in _local_traces:
        _local_traces[trace_id]["output"] = output

    root_span = _root_spans.pop(trace_id, None)
    if _enabled and root_span is not None:
        try:
            root_span.update(output=output)
            root_span.end()
            _client.flush()
        except Exception as exc:  # pragma: no cover - network/API-shape failure
            logger.warning("Langfuse end_trace failed for trace %s (%s).", trace_id, exc)
    elif not _enabled:
        logger.info("[trace:end] trace_id=%s output=%s", trace_id, output)


def get_trace(trace_id: str) -> Optional[dict]:
    return _local_traces.get(trace_id)


def get_trace_url(trace_id: str) -> Optional[str]:
    """The actual Langfuse dashboard URL for a trace, when enabled — this is
    what failure analysis should link to, not just the raw trace_id."""
    if not _enabled:
        return None
    try:
        return _client.get_trace_url(trace_id=trace_id)
    except Exception:  # pragma: no cover
        return None


def is_langfuse_enabled() -> bool:
    return _enabled
