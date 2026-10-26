"""Probabilistic power-system forecasts with a frozen foundation model."""
from importlib import import_module

__version__ = "0.3.1"

for _module in ("torch", "torch_geometric", "torch_scatter"):
    try:
        import_module(_module)
    except (ImportError, OSError) as _error:
        raise ImportError(
            f"power-fm-foresight requires a compatible torch/PyG/scatter stack; "
            f"could not import {_module}. See README.md#install-matrix. "
            "Install the stack for your hardware before this package; preserve Aurora vendor torch."
        ) from _error
del _module
