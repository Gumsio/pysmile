"""Check that installed public imports use the same package implementations."""

from importlib.metadata import metadata

import pysmile
from pysmile import calib, prep, stats, plot


def test_public_imports_match_package_modules():
    for module in [calib, prep, stats, plot]:
        for name in pysmile.__all__:
            if hasattr(module, name):
                assert getattr(pysmile, name) is getattr(module, name)
    assert len(pysmile.__all__) == 11


def test_installed_metadata_declares_runtime_dependencies():
    project = metadata("pysmile")
    assert project["Version"] == "0.1.0.dev0"
    for dependency in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]:
        assert any(requirement.startswith(dependency + ">=")
                   for requirement in project.get_all("Requires-Dist"))
    assert "test" in project.get_all("Provides-Extra")
