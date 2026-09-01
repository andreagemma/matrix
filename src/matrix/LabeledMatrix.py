"""Generic labeled 2D matrices backed by NumPy arrays."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

Label: TypeAlias = Hashable
Labels: TypeAlias = Sequence[Label]
IndexLike: TypeAlias = Label | Sequence[Label] | slice | None


class _AxisIndexer:
    """Internal helper implementing ``.loc`` and ``.iloc`` style indexing."""

    def __init__(self, parent: LabeledMatrix, use_labels: bool) -> None:
        self._parent = parent
        self._use_labels = use_labels

    def __getitem__(self, key: IndexLike | tuple[IndexLike, IndexLike]) -> Any:
        if not isinstance(key, tuple):
            raise TypeError("2D indexing expected: use [rows, cols]")
        row_key, col_key = key
        row_index = self._parent._resolve_axis_key(
            row_key,
            axis=0,
            by_label=self._use_labels,
        )
        col_index = self._parent._resolve_axis_key(
            col_key,
            axis=1,
            by_label=self._use_labels,
        )
        values = self._parent._values
        if isinstance(row_index, int) and isinstance(col_index, int):
            item = values[row_index, col_index]
            return item.item() if values.dtype.kind in "iufb" else item

        scalar_row = isinstance(row_index, int)
        scalar_col = isinstance(col_index, int)
        row_index = np.array([row_index]) if scalar_row else np.asarray(row_index)
        col_index = np.array([col_index]) if scalar_col else np.asarray(col_index)
        output = values[np.ix_(row_index, col_index)]
        if scalar_row and not scalar_col:
            return output[0, :]
        if scalar_col and not scalar_row:
            return output[:, 0]
        return output

    def __setitem__(self, key: IndexLike | tuple[IndexLike, IndexLike], value: Any) -> None:
        if not isinstance(key, tuple):
            raise TypeError("2D indexing expected: use [rows, cols]")
        row_key, col_key = key
        row_index = self._parent._resolve_axis_key(
            row_key,
            axis=0,
            by_label=self._use_labels,
        )
        col_index = self._parent._resolve_axis_key(
            col_key,
            axis=1,
            by_label=self._use_labels,
        )
        if isinstance(row_index, int) and isinstance(col_index, int):
            self._parent._values[row_index, col_index] = value
            return
        row_index = np.array([row_index]) if isinstance(row_index, int) else np.asarray(row_index)
        col_index = np.array([col_index]) if isinstance(col_index, int) else np.asarray(col_index)
        self._parent._values[np.ix_(row_index, col_index)] = value


class LabeledMatrix:
    """A compact 2D array with hashable row and column labels.

    Data are stored in a contiguous NumPy array for speed and memory efficiency,
    while dictionaries map labels to integer positions for O(1) lookup.
    """

    __slots__ = ("_values", "_row_index", "_col_index", "_row_pos", "_col_pos")

    def __init__(
        self,
        data: Sequence[Sequence[Any]] | np.ndarray,
        *,
        row_index: Labels,
        col_index: Labels,
        dtype: np.dtype | None = None,
        copy: bool = False,
    ) -> None:
        array = np.asarray(data, dtype=dtype)
        array = array.copy(order="C") if copy else np.ascontiguousarray(array)
        if array.ndim != 2:
            raise ValueError("data must be 2D")
        row_count, col_count = array.shape
        if len(row_index) != row_count:
            raise ValueError("row_index length does not match number of rows")
        if len(col_index) != col_count:
            raise ValueError("col_index length does not match number of columns")
        if len(set(row_index)) != len(row_index):
            raise ValueError("row_index labels must be unique")
        if len(set(col_index)) != len(col_index):
            raise ValueError("col_index labels must be unique")

        self._values = array
        self._row_index = tuple(row_index)
        self._col_index = tuple(col_index)
        self._row_pos = {label: index for index, label in enumerate(self._row_index)}
        self._col_pos = {label: index for index, label in enumerate(self._col_index)}

    @property
    def shape(self) -> tuple[int, int]:
        """Return the ``(rows, columns)`` shape of the matrix."""
        return self._values.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the NumPy dtype of the matrix values."""
        return self._values.dtype

    @property
    def values(self) -> np.ndarray:
        """Return a view of the underlying NumPy array."""
        return self._values

    @property
    def row_index(self) -> tuple[Label, ...]:
        """Return immutable row labels."""
        return self._row_index

    @property
    def col_index(self) -> tuple[Label, ...]:
        """Return immutable column labels."""
        return self._col_index

    def __repr__(self) -> str:
        rows, cols = self.shape
        return (
            f"LabeledMatrix(shape={rows}x{cols}, dtype={self.dtype}, "
            f"rows={list(self._row_index)[:3]}{'...' if rows > 3 else ''}, "
            f"cols={list(self._col_index)[:3]}{'...' if cols > 3 else ''})"
        )

    def _resolve_axis_key(self, key: IndexLike, *, axis: int, by_label: bool) -> Any:
        axis_size = self.shape[axis]
        if key is None:
            return np.arange(axis_size)
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = axis_size if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            return np.arange(axis_size)[slice(start, stop, step)]
        if by_label:
            mapper = self._row_pos if axis == 0 else self._col_pos
            if isinstance(key, (list, tuple)):
                return [mapper[self._ensure_label(label)] for label in key]
            return mapper[self._ensure_label(key)]
        if isinstance(key, (list, tuple, np.ndarray)):
            return key
        if isinstance(key, int):
            if key < 0 or key >= axis_size:
                raise IndexError("positional index out of range")
            return key
        raise TypeError("Invalid index type")

    @staticmethod
    def _ensure_label(value: Any) -> Label:
        try:
            hash(value)
        except Exception as exc:  # pragma: no cover
            raise TypeError("Labels must be hashable") from exc
        return value

    def __getitem__(self, key: tuple[Any, Any]) -> Any:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Use M[row_label, col_label] or M[i, j]")
        row_key, col_key = key
        by_label_row = row_key in self._row_pos if isinstance(row_key, Hashable) else False
        by_label_col = col_key in self._col_pos if isinstance(col_key, Hashable) else False
        row_index = self._resolve_axis_key(row_key, axis=0, by_label=by_label_row)
        col_index = self._resolve_axis_key(col_key, axis=1, by_label=by_label_col)
        return self._values[row_index, col_index]

    def __setitem__(self, key: tuple[Any, Any], value: Any) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Use M[row_label, col_label] or M[i, j]")
        row_key, col_key = key
        by_label_row = row_key in self._row_pos if isinstance(row_key, Hashable) else False
        by_label_col = col_key in self._col_pos if isinstance(col_key, Hashable) else False
        row_index = self._resolve_axis_key(row_key, axis=0, by_label=by_label_row)
        col_index = self._resolve_axis_key(col_key, axis=1, by_label=by_label_col)
        self._values[row_index, col_index] = value

    @property
    def loc(self) -> _AxisIndexer:
        """Return a label-based indexer."""
        return _AxisIndexer(self, use_labels=True)

    @property
    def iloc(self) -> _AxisIndexer:
        """Return a position-based indexer."""
        return _AxisIndexer(self, use_labels=False)

    @classmethod
    def from_dict_of_dicts(
        cls,
        data: Mapping[Label, Mapping[Label, Any]],
        *,
        dtype: np.dtype | None = None,
        row_order: Labels | None = None,
        col_order: Labels | None = None,
        fill_value: Any = 0,
    ) -> LabeledMatrix:
        """Create a matrix from nested dictionaries ``data[row][col] -> value``."""
        rows = list(row_order) if row_order is not None else list(data.keys())
        cols = (
            list(col_order)
            if col_order is not None
            else sorted({column for values in data.values() for column in values}, key=str)
        )
        matrix = np.full(
            (len(rows), len(cols)),
            fill_value,
            dtype=dtype if dtype is not None else float,
        )
        row_pos = {row: index for index, row in enumerate(rows)}
        col_pos = {col: index for index, col in enumerate(cols)}
        for row, values in data.items():
            for col, value in values.items():
                matrix[row_pos[row], col_pos[col]] = value
        return cls(matrix, row_index=rows, col_index=cols, dtype=matrix.dtype)

    def reindex(
        self,
        *,
        rows: Labels | None = None,
        cols: Labels | None = None,
        fill_value: Any = 0,
    ) -> LabeledMatrix:
        """Return a copy aligned to new row or column labels."""
        rows = self._row_index if rows is None else tuple(rows)
        cols = self._col_index if cols is None else tuple(cols)
        output = np.full((len(rows), len(cols)), fill_value, dtype=self._values.dtype)
        row_pos = {row: index for index, row in enumerate(rows)}
        col_pos = {col: index for index, col in enumerate(cols)}
        for old_row_index, row in enumerate(self._row_index):
            if row not in row_pos:
                continue
            for old_col_index, col in enumerate(self._col_index):
                if col not in col_pos:
                    continue
                output[row_pos[row], col_pos[col]] = self._values[old_row_index, old_col_index]
        return LabeledMatrix(output, row_index=rows, col_index=cols, dtype=output.dtype)

    def rename(
        self,
        *,
        rows: Mapping[Label, Label] | None = None,
        cols: Mapping[Label, Label] | None = None,
    ) -> LabeledMatrix:
        """Return a copy with renamed labels."""
        new_rows = (
            [rows.get(row, row) for row in self._row_index] if rows else list(self._row_index)
        )
        new_cols = (
            [cols.get(col, col) for col in self._col_index] if cols else list(self._col_index)
        )
        if len(set(new_rows)) != len(new_rows):
            raise ValueError("row renaming introduces duplicates")
        if len(set(new_cols)) != len(new_cols):
            raise ValueError("col renaming introduces duplicates")
        return LabeledMatrix(
            self._values.copy(order="C"),
            row_index=new_rows,
            col_index=new_cols,
            dtype=self._values.dtype,
        )

    def _binary_op(self, other: Any, op: Any) -> LabeledMatrix:
        if isinstance(other, LabeledMatrix):
            if self.row_index != other.row_index or self.col_index != other.col_index:
                raise ValueError(
                    "Labels must align for element-wise operations. Use .reindex first."
                )
            output = op(self._values, other._values)
        else:
            output = op(self._values, other)
        return LabeledMatrix(
            output,
            row_index=self._row_index,
            col_index=self._col_index,
            dtype=output.dtype,
        )

    def __add__(self, other: Any) -> LabeledMatrix:
        """Return element-wise addition with a scalar or aligned matrix."""
        return self._binary_op(other, np.add)

    def __sub__(self, other: Any) -> LabeledMatrix:
        """Return element-wise subtraction with a scalar or aligned matrix."""
        return self._binary_op(other, np.subtract)

    def __mul__(self, other: Any) -> LabeledMatrix:
        """Return element-wise multiplication with a scalar or aligned matrix."""
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other: Any) -> LabeledMatrix:
        """Return element-wise division with a scalar or aligned matrix."""
        return self._binary_op(other, np.divide)

    def tolist(self) -> list[list[Any]]:
        """Return the values as nested Python lists."""
        return self._values.tolist()

    def to_numpy(self, *, copy: bool = False) -> np.ndarray:
        """Return the underlying NumPy array or a C-contiguous copy."""
        return self._values.copy(order="C") if copy else self._values

    def to_pandas(self) -> Any:
        """Return the matrix as a pandas DataFrame."""
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is not installed") from exc
        return pd.DataFrame(self._values, index=self._row_index, columns=self._col_index)


__all__ = ["LabeledMatrix"]
