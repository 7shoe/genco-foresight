"""Compatibility aliases for paths recorded in historical run manifests."""
from importlib import import_module
import runpy
import sys
import warnings


def reexport(old, new):
    warnings.warn(f"{old} is deprecated; use {new}", DeprecationWarning, stacklevel=2)
    if old == "__main__":
        runpy.run_module(new, run_name="__main__")
    else:
        sys.modules[old] = import_module(new)
