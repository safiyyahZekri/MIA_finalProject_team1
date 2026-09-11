"""
Guards against the exact bug found during verification: this module was
originally written against Langfuse's older `.trace()` / `.span()` API,
which does not exist on the SDK version actually installed
(`langfuse==4.15.1`, OpenTelemetry-based client). Calling the real SDK with
real credentials raised `AttributeError: 'Langfuse' object has no attribute
'trace'` immediately.

These tests monkeypatch `tracing._client` / `tracing._enabled` with a
stand-in that mirrors the *real* v4 API surface (`create_trace_id`,
`start_observation`, span `.update()` / `.end()`, `.flush()`,
`get_trace_url`) so a future regression back to the old API shape fails
here instead of silently passing (since the local-fallback path never
exercises `_client` at all and would hide such a bug).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tracing  # noqa: E402


class FakeSpan:
    def __init__(self, name, input=None, output=None, metadata=None):
        self.name = name
        self.input = input
        self.output = output
        self.metadata = metadata
        self.ended = False
        self.children = []

    def start_observation(self, *, name, input=None, output=None, metadata=None, **kw):
        child = FakeSpan(name, input, output, metadata)
        self.children.append(child)
        return child

    def update(self, *, output=None, **kw):
        self.output = output
        return self

    def end(self, **kw):
        self.ended = True
        return self


class FakeLangfuseClient:
    """Mirrors only the methods tracing.py actually calls on a real
    `langfuse.Langfuse` v4 client instance."""

    def __init__(self):
        self.flushed = False
        self.roots = {}

    @staticmethod
    def create_trace_id(seed=None):
        return "fake-otel-trace-id-0123456789abcdef"

    def start_observation(self, *, trace_context, name, as_type="span", metadata=None, **kw):
        span = FakeSpan(name, metadata=metadata)
        self.roots[trace_context["trace_id"]] = span
        return span

    def flush(self):
        self.flushed = True

    def get_trace_url(self, *, trace_id):
        return f"https://fake-langfuse.example/trace/{trace_id}"


def test_start_log_end_against_real_api_shape(monkeypatch):
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(tracing, "_client", fake_client)
    monkeypatch.setattr(tracing, "_enabled", True)
    monkeypatch.setattr(tracing, "_root_spans", {})

    trace_id = tracing.start_trace("unit-test-trace", metadata={"k": "v"})
    assert trace_id in fake_client.roots

    tracing.log_step(trace_id, "step1", input={"q": "hi"}, output={"a": "ok"}, latency_ms=10.0)
    root_span = fake_client.roots[trace_id]
    assert len(root_span.children) == 1
    assert root_span.children[0].name == "step1"
    assert root_span.children[0].ended is True

    tracing.end_trace(trace_id, output={"done": True})
    assert root_span.ended is True
    assert root_span.output == {"done": True}
    assert fake_client.flushed is True

    url = tracing.get_trace_url(trace_id)
    assert url == f"https://fake-langfuse.example/trace/{trace_id}"

    # Local fallback view must also reflect everything, even when Langfuse
    # is enabled — it's still used for the /trace/{id} inspection endpoint.
    local = tracing.get_trace(trace_id)
    assert local["steps"][0]["name"] == "step1"
    assert local["output"] == {"done": True}


def test_tracing_never_raises_when_langfuse_api_call_fails(monkeypatch):
    """If the SDK's API shape ever changes again, tracing must degrade
    (log + continue), not crash the caller's request."""

    class BrokenClient:
        @staticmethod
        def create_trace_id(seed=None):
            return "broken-trace-id"

        def start_observation(self, **kw):
            raise AttributeError("simulated API-shape mismatch")

        def flush(self):
            raise AttributeError("simulated API-shape mismatch")

        def get_trace_url(self, **kw):
            raise AttributeError("simulated API-shape mismatch")

    monkeypatch.setattr(tracing, "_client", BrokenClient())
    monkeypatch.setattr(tracing, "_enabled", True)
    monkeypatch.setattr(tracing, "_root_spans", {})

    trace_id = tracing.start_trace("will-fail", metadata={})
    tracing.log_step(trace_id, "step", input=None, output=None)
    tracing.end_trace(trace_id, output=None)
    assert tracing.get_trace_url(trace_id) is None
    # No exception propagated, and the local record still works.
    assert tracing.get_trace(trace_id) is not None
