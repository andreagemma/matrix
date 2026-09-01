"""Two-dimensional origin-destination matrices with row and column labels."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from os import PathLike
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

Label: TypeAlias = Hashable
LabelMap: TypeAlias = Mapping[Label, int]
LabelsInput: TypeAlias = Sequence[Label] | np.ndarray | LabelMap
MatrixInit: TypeAlias = Mapping[Label, Mapping[Label, float]] | Sequence[Sequence[float]]


def _convert_to_dict(labels: LabelsInput) -> LabelMap:
    """Convert labels or an existing label-position mapping to an immutable mapping."""
    if isinstance(labels, Mapping):
        converted_mapping: dict[Label, int] = dict(labels)
        positions = list(converted_mapping.values())
        if not all(isinstance(position, int) for position in positions):
            raise ValueError("Label mapping values must be integer positions.")
        if sorted(positions) != list(range(len(converted_mapping))):
            raise ValueError("Label mapping positions must be unique and zero-based.")
        return MappingProxyType(converted_mapping)

    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise TypeError("Labels must be a sequence, NumPy array, or mapping.")

    converted_labels: dict[Label, int] = {}
    for index, label in enumerate(labels):
        if label in converted_labels:
            raise ValueError(f"Duplicate label: {label!r}")
        converted_labels[label] = index
    return MappingProxyType(converted_labels)


class MatrixOD:
    """A NumPy-backed matrix addressed by origin and destination labels.

    Parameters
    ----------
    rows:
        Row labels or a label-to-position mapping.
    cols:
        Column labels or a label-to-position mapping.
    init:
        Optional initial data. Accepts nested dictionaries, rectangular sequences,
        scalar values, NumPy arrays, or another ``MatrixOD``.
    copy:
        Whether to copy NumPy or ``MatrixOD`` data instead of sharing it.
    mode:
        Optional user metadata preserved on copies and arithmetic results.
    """

    def __init__(
        self,
        rows: LabelsInput,
        cols: LabelsInput,
        init: MatrixInit | int | float | np.ndarray | MatrixOD | None = None,
        copy: bool = False,
        mode: str | None = None,
    ) -> None:
        self.rows = _convert_to_dict(rows)
        self.cols = _convert_to_dict(cols)
        self.mode = mode
        self.mat = self._init_array(init, copy=copy)

    def _init_array(
        self,
        init: MatrixInit | int | float | np.ndarray | MatrixOD | None,
        *,
        copy: bool,
    ) -> np.ndarray:
        shape = (len(self.rows), len(self.cols))
        if init is None:
            return np.zeros(shape, dtype=float)
        if isinstance(init, MatrixOD):
            array = init.mat.copy() if copy else init.mat
        elif isinstance(init, Mapping):
            array = np.zeros(shape, dtype=float)
            for row_label, row_values in init.items():
                for col_label, value in row_values.items():
                    row_index = self._row_position(row_label)
                    col_index = self._col_position(col_label)
                    array[row_index, col_index] = value
        elif isinstance(init, np.ndarray):
            array = init.copy() if copy else init
        elif isinstance(init, (int, float)):
            array = np.full(shape, init, dtype=float)
        elif isinstance(init, Sequence) and not isinstance(init, (str, bytes)):
            array = np.asarray(init, dtype=float)
        else:
            raise TypeError("Unsupported type for init.")

        if array.shape != shape:
            raise ValueError(
                f"Initial data shape {array.shape} does not match matrix shape {shape}."
            )
        return array

    def _row_position(self, label: Label) -> int:
        try:
            return self.rows[label]
        except KeyError as exc:
            raise KeyError(f"Row label {label!r} not found.") from exc

    def _col_position(self, label: Label) -> int:
        try:
            return self.cols[label]
        except KeyError as exc:
            raise KeyError(f"Column label {label!r} not found.") from exc

    def _ensure_same_labels(self, other: MatrixOD) -> None:
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same row and column labels.")

    def copy(self, copy_data: bool = True) -> MatrixOD:
        """Return a copy of this matrix, optionally sharing the underlying array."""
        return MatrixOD(
            self.rows,
            self.cols,
            init=self.mat,
            copy=copy_data,
            mode=self.mode,
        )

    def __getitem__(self, pos: tuple[Label, Label]) -> float:
        row_label, col_label = pos
        return self.mat[self._row_position(row_label), self._col_position(col_label)]

    def __setitem__(self, pos: tuple[Label, Label], value: float) -> None:
        row_label, col_label = pos
        self.mat[self._row_position(row_label), self._col_position(col_label)] = value

    def __repr__(self) -> str:
        return repr(self.mat)

    def __str__(self) -> str:
        row_labels = list(self.rows.keys())
        col_labels = list(self.cols.keys())

        if len(row_labels) > 10:
            row_labels = row_labels[:5] + ["..."] + row_labels[-5:]
        if len(col_labels) > 10:
            col_labels = col_labels[:5] + ["..."] + col_labels[-5:]

        header = "     " + " ".join(f"{col!s:>8}" for col in col_labels) + "\n"
        rows_str = ""
        for row in row_labels:
            if row == "...":
                rows_str += f"{row:>4} {'...':>8} {'...':>8} {'...':>8}\n"
                continue
            row_data = " ".join(
                f"{self[row, col]:>8.2f}" if col != "..." else "..." for col in col_labels
            )
            rows_str += f"{row!s:>4} {row_data}\n"
        return header + rows_str

    def __neg__(self) -> MatrixOD:
        return MatrixOD(self.rows, self.cols, init=-self.mat, mode=self.mode)

    def __add__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            return MatrixOD(self.rows, self.cols, init=self.mat + other.mat, mode=self.mode)
        if isinstance(other, (int, float)):
            return MatrixOD(self.rows, self.cols, init=self.mat + other, mode=self.mode)
        raise TypeError("Unsupported operand type for addition.")

    __radd__ = __add__

    def __sub__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            return MatrixOD(self.rows, self.cols, init=self.mat - other.mat, mode=self.mode)
        if isinstance(other, (int, float)):
            return MatrixOD(self.rows, self.cols, init=self.mat - other, mode=self.mode)
        raise TypeError("Unsupported operand type for subtraction.")

    def __rsub__(self, other: int | float) -> MatrixOD:
        if isinstance(other, (int, float)):
            return MatrixOD(self.rows, self.cols, init=other - self.mat, mode=self.mode)
        raise TypeError("Unsupported operand type for subtraction.")

    def __iadd__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            self.mat += other.mat
        elif isinstance(other, (int, float)):
            self.mat += other
        else:
            raise TypeError("Unsupported operand type for addition.")
        return self

    def __isub__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            self.mat -= other.mat
        elif isinstance(other, (int, float)):
            self.mat -= other
        else:
            raise TypeError("Unsupported operand type for subtraction.")
        return self

    def __mul__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            return MatrixOD(self.rows, self.cols, init=self.mat * other.mat, mode=self.mode)
        if isinstance(other, (int, float)):
            return MatrixOD(self.rows, self.cols, init=self.mat * other, mode=self.mode)
        raise TypeError("Unsupported operand type for multiplication.")

    __rmul__ = __mul__

    def __imul__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            self.mat *= other.mat
        elif isinstance(other, (int, float)):
            self.mat *= other
        else:
            raise TypeError("Unsupported operand type for multiplication.")
        return self

    def __truediv__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            return MatrixOD(self.rows, self.cols, init=self.mat / other.mat, mode=self.mode)
        if isinstance(other, (int, float)):
            return MatrixOD(self.rows, self.cols, init=self.mat / other, mode=self.mode)
        raise TypeError("Unsupported operand type for division.")

    def __rtruediv__(self, other: int | float) -> MatrixOD:
        if isinstance(other, (int, float)):
            return MatrixOD(self.rows, self.cols, init=other / self.mat, mode=self.mode)
        raise TypeError("Unsupported operand type for division.")

    def __itruediv__(self, other: int | float | MatrixOD) -> MatrixOD:
        if isinstance(other, MatrixOD):
            self._ensure_same_labels(other)
            self.mat /= other.mat
        elif isinstance(other, (int, float)):
            self.mat /= other
        else:
            raise TypeError("Unsupported operand type for division.")
        return self

    def transpose(self) -> MatrixOD:
        """Return a transposed matrix with rows and columns swapped."""
        return MatrixOD(self.cols, self.rows, init=self.mat.T, mode=self.mode)

    def inverse(self) -> MatrixOD:
        """Return the inverse of a square matrix."""
        if self.mat.shape[0] != self.mat.shape[1]:
            raise ValueError("Matrix must be square to find its inverse.")
        return MatrixOD(self.rows, self.cols, init=np.linalg.inv(self.mat), mode=self.mode)

    def get_diagonal(self) -> np.ndarray:
        """Return the main diagonal values."""
        return np.diag(self.mat)

    def set_diagonal(self, values: Sequence[float]) -> None:
        """Replace the main diagonal values."""
        if len(values) != min(self.mat.shape):
            raise ValueError("Length of values must match the length of the matrix diagonal.")
        np.fill_diagonal(self.mat, values)

    def nan_to_num(
        self,
        copy: bool = True,
        nan: float = 0.0,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> None:
        """Replace NaN and infinite values in the matrix."""
        self.mat = np.nan_to_num(self.mat, copy=copy, nan=nan, posinf=posinf, neginf=neginf)

    def sum(self, axis: int | None = None) -> float | MatrixOD:
        """Sum all values, rows, or columns.

        ``axis=None`` returns a scalar. ``axis=0`` returns one ``sum`` row with
        column totals. ``axis=1`` returns one ``sum`` column with row totals.
        """
        if axis is None:
            return float(np.sum(self.mat))
        if axis == 0:
            summed_cols = np.sum(self.mat, axis=0)[np.newaxis, :]
            return MatrixOD(["sum"], self.cols, init=summed_cols, mode=self.mode)
        if axis == 1:
            summed_rows = np.sum(self.mat, axis=1)[:, np.newaxis]
            return MatrixOD(self.rows, ["sum"], init=summed_rows, mode=self.mode)
        raise ValueError("Axis must be 0, 1, or None.")

    @staticmethod
    def read_df(
        rows: LabelsInput,
        cols: LabelsInput,
        df: pd.DataFrame,
        o_field: str = "o",
        d_field: str = "d",
        value_field: str = "value",
    ) -> MatrixOD:
        """Create a matrix from a long-form DataFrame."""
        matrix = MatrixOD(rows=rows, cols=cols)
        frame = df[[o_field, d_field, value_field]].rename(
            columns={o_field: "o", d_field: "d", value_field: "value"}
        )
        for origin, destination, value in frame.itertuples(index=False, name=None):
            matrix[origin, destination] = value
        return matrix

    @staticmethod
    def read_csv(
        rows: LabelsInput,
        cols: LabelsInput,
        file: str | PathLike[str],
        o_field: str = "o",
        d_field: str = "d",
        value_field: str = "value",
    ) -> MatrixOD:
        """Create a matrix from a long-form CSV file."""
        df = pd.read_csv(file, usecols=[o_field, d_field, value_field])
        return MatrixOD.read_df(
            rows=rows,
            cols=cols,
            df=df,
            o_field=o_field,
            d_field=d_field,
            value_field=value_field,
        )

    def write_df(
        self,
        o_field: str = "o",
        d_field: str = "d",
        value_field: str = "value",
    ) -> pd.DataFrame:
        """Return the matrix as a long-form DataFrame."""
        data: list[dict[str, Any]] = []
        for origin, origin_index in self.rows.items():
            for destination, destination_index in self.cols.items():
                data.append(
                    {
                        o_field: origin,
                        d_field: destination,
                        value_field: self.mat[origin_index, destination_index],
                    }
                )
        return pd.DataFrame(data)

    def write_csv(
        self,
        file: str | PathLike[str],
        o_field: str = "o",
        d_field: str = "d",
        value_field: str = "value",
    ) -> None:
        """Write the matrix as a long-form CSV file."""
        df = self.write_df(o_field=o_field, d_field=d_field, value_field=value_field)
        df.to_csv(file, index=False)
