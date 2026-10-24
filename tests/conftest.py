"""Use a noninteractive backend and a writable font cache during tests."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pysmile-test-matplotlib"))

import matplotlib

matplotlib.use("Agg")
