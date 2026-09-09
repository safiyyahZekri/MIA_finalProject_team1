"""Make `app` importable no matter how pytest is invoked.

Each test module here inserts the service root into sys.path itself, which
works only for whichever module is imported first -- and collection runs in
alphabetical order, so a new test file sorting before the one carrying the
insert fails at import with `No module named 'app'`. It also passes locally
under `python -m pytest`, which puts the working directory on sys.path, and
fails under bare `pytest`, which does not -- exactly how CI runs it.

conftest.py is imported before any test module, so doing it once here holds
for every file regardless of name or invocation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
