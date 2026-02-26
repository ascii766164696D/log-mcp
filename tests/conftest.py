from __future__ import annotations

import pytest
import tempfile
import os


@pytest.fixture
def tmp_log(tmp_path):
    """Factory fixture that creates a temp log file with given content."""
    def _make(content: str, name: str = "test.log") -> str:
        p = tmp_path / name
        p.write_text(content)
        return str(p)
    return _make
