"""GA Matrix: label-aware matrix utilities."""

from ._version import __version__
from .LabeledMatrix import LabeledMatrix
from .matrix_od import MatrixOD
from .matrix_odt import MatrixODT

__all__ = ["LabeledMatrix", "MatrixOD", "MatrixODT", "__version__"]
