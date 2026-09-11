"""Make `app` importable, and pin the test environment, before any test
module is imported -- no matter how pytest is invoked.

Each test module used to insert the service root into sys.path itself, which
works only for whichever module is imported first. Collection runs in
alphabetical order, so a new test file sorting before the one carrying the
insert failed with `No module named 'app'`. It also passed locally under
`python -m pytest`, which puts the working directory on sys.path, and failed
under bare `pytest`, which does not -- exactly how CI runs it.

The environment defaults live here for the same reason. app.config reads
os.environ once, at first import: that is whichever test module sorts first,
not test_graph.py where these defaults were originally set. It also calls
load_dotenv(), which searches parent directories and would pick up a local
.env holding real keys. With retrieval-api listening on localhost:8002 the
graph tests would otherwise query a live index instead of the offline
fallback they assert against.
"""

import os
import sys

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("RETRIEVAL_API_URL", "http://localhost:19999")  # unreachable on purpose
os.environ.setdefault("RETRIEVAL_FALLBACK_TO_MOCK", "true")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
