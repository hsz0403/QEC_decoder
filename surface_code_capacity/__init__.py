"""Surface-code code-capacity dataset and CNN decoder utilities."""

from .data import (
    build_surface_code_iid_dataset,
    load_surface_code_dataset,
    save_surface_code_dataset,
)
from .model import SurfaceCodeCNNDecoder

__all__ = [
    "SurfaceCodeCNNDecoder",
    "build_surface_code_iid_dataset",
    "load_surface_code_dataset",
    "save_surface_code_dataset",
]
