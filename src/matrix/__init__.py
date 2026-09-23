"""GA Matrix: label-aware matrix utilities."""

from ._version import __version__
from .LabeledMatrix import LabeledMatrix
from .matrix_od import MatrixOD, LabelsInput, MatrixInit
from .matrix_odt import MatrixODT, Timestamp

__all__ = ["LabeledMatrix", 
           "MatrixOD", 
           "MatrixODT", 
           "__version__", 
           "Timestamp", 
           "LabelsInput", 
           "MatrixInit"] 
