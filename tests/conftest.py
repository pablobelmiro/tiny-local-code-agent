"""Test-session defaults so importing neuralcode.config never KeyErrors.

config.py requires BASE_URL/API_KEY to be set (real usage always has them).
Individual tests that care about specific values still set them explicitly
via monkeypatch, which takes precedence and is undone after each test.
"""

import os

os.environ.setdefault("BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("API_KEY", "test-key")
