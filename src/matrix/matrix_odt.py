"""Time-indexed origin-destination matrices."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from os import PathLike
from typing import Any, TypeAlias

import pandas as pd

from .matrix_od import LabelMap, LabelsInput, MatrixInit, MatrixOD, _convert_to_dict

Timestamp: TypeAlias = Hashable
ODTInit: TypeAlias = Mapping[Timestamp, MatrixOD | MatrixInit | int | float]


class MatrixODT:
    """A collection of ``MatrixOD`` instances indexed by timestamp labels.

    Missing timestamps are treated as zero matrices when reading values. Assigning
    a timestamp creates or replaces the corresponding ``MatrixOD``.
    """

    def __init__(
        self,
        rows: LabelsInput,
        cols: LabelsInput,
        timestamps: list[Timestamp] | set[Timestamp] | tuple[Timestamp, ...],
        init: ODTInit | MatrixODT | None = None,
        copy: bool = False,
        mode: str | None = None,
    ) -> None:
        self.rows: LabelMap
        self.cols: LabelMap
        self.timestamps: set[Timestamp]
        self.mode: str | None
        self.ods: dict[Timestamp, MatrixOD]

        if isinstance(init, MatrixODT):
            self.rows = init.rows
            self.cols = init.cols
            self.timestamps = set(init.timestamps)
            self.mode = init.mode if mode is None else mode
            self.ods = {
                timestamp: matrix.copy(copy_data=True) if copy else matrix
                for timestamp, matrix in init.ods.items()
            }
            return

        self.rows = _convert_to_dict(rows)
        self.cols = _convert_to_dict(cols)
        self.timestamps = set(timestamps)
        self.mode = mode
        self.ods = {
            timestamp: MatrixOD(self.rows, self.cols, mode=mode) for timestamp in self.timestamps
        }

        if init is not None:
            for timestamp, value in init.items():
                self.timestamps.add(timestamp)
                self.ods[timestamp] = self._coerce_matrix(value, copy=copy)

    def _coerce_matrix(
        self,
        value: MatrixOD | MatrixInit | int | float,
        *,
        copy: bool,
    ) -> MatrixOD:
        if isinstance(value, MatrixOD):
            if value.rows != self.rows or value.cols != self.cols:
                raise ValueError("Matrices must have the same row and column labels.")
            return value.copy(copy_data=True) if copy else value
        return MatrixOD(self.rows, self.cols, init=value, copy=copy, mode=self.mode)

    def _zero_matrix(self) -> MatrixOD:
        return MatrixOD(self.rows, self.cols, mode=self.mode)

    def _matrix_or_zero(self, timestamp: Timestamp) -> MatrixOD:
        return self.ods.get(timestamp, self._zero_matrix())

    def _ensure_same_axes(self, other: MatrixODT) -> None:
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same row and column labels.")

    def copy(self, copy_data: bool = True) -> MatrixODT:
        """Return a copy of this time-indexed matrix."""
        return MatrixODT(
            self.rows,
            self.cols,
            self.timestamps,
            init=self,
            copy=copy_data,
            mode=self.mode,
        )

    def __getitem__(
        self,
        pos: Timestamp | tuple[Hashable, Hashable, Timestamp],
    ) -> MatrixOD | float:
        if not isinstance(pos, tuple):
            return self._matrix_or_zero(pos)
        if len(pos) != 3:
            raise TypeError("Use matrix[o, d, timestamp] for scalar access.")
        origin, destination, timestamp = pos
        if timestamp not in self.ods:
            return 0.0
        return self.ods[timestamp][origin, destination]

    def __setitem__(
        self,
        pos: Timestamp | tuple[Hashable, Hashable, Timestamp],
        value: MatrixOD | MatrixInit | int | float,
    ) -> None:
        if not isinstance(pos, tuple):
            self.timestamps.add(pos)
            self.ods[pos] = self._coerce_matrix(value, copy=False)
            return
        if len(pos) != 3:
            raise TypeError("Use matrix[o, d, timestamp] for scalar access.")
        origin, destination, timestamp = pos
        if timestamp not in self.ods:
            self.timestamps.add(timestamp)
            self.ods[timestamp] = self._zero_matrix()
        if not isinstance(value, (int, float)):
            raise TypeError("Scalar assignment requires an int or float value.")
        self.ods[timestamp][origin, destination] = value

    def sum(self, axis: int | None = None) -> float | MatrixOD | MatrixODT:
        """Sum all values or collapse one axis.

        ``axis=None`` returns a scalar. ``axis=0`` and ``axis=1`` preserve the
        timestamp dimension. ``axis=2`` sums all timestamps into one ``MatrixOD``.
        """
        if axis is None:
            return sum(matrix.sum() for matrix in self.ods.values())
        if axis == 0:
            summed_cols = MatrixODT(["sum"], self.cols, self.timestamps, mode=self.mode)
            for timestamp, matrix in self.ods.items():
                summed_cols[timestamp] = matrix.sum(axis=0)
            return summed_cols
        if axis == 1:
            summed_rows = MatrixODT(self.rows, ["sum"], self.timestamps, mode=self.mode)
            for timestamp, matrix in self.ods.items():
                summed_rows[timestamp] = matrix.sum(axis=1)
            return summed_rows
        if axis == 2:
            summed_timestamps = MatrixOD(self.rows, self.cols, mode=self.mode)
            for matrix in self.ods.values():
                summed_timestamps += matrix
            return summed_timestamps
        raise ValueError("Axis must be 0, 1, 2, or None.")

    def __add__(self, other: int | float | MatrixODT) -> MatrixODT:
        result = self.copy()
        result += other
        return result

    def __iadd__(self, other: int | float | MatrixODT) -> MatrixODT:
        if isinstance(other, MatrixODT):
            self._ensure_same_axes(other)
            timestamps = self.timestamps | other.timestamps
            self.ods = {
                timestamp: self._matrix_or_zero(timestamp) + other._matrix_or_zero(timestamp)
                for timestamp in timestamps
            }
            self.timestamps = timestamps
        elif isinstance(other, (int, float)):
            for matrix in self.ods.values():
                matrix += other
        else:
            raise TypeError("Unsupported operand type for addition.")
        return self

    def __sub__(self, other: int | float | MatrixODT) -> MatrixODT:
        result = self.copy()
        result -= other
        return result

    def __isub__(self, other: int | float | MatrixODT) -> MatrixODT:
        if isinstance(other, MatrixODT):
            self._ensure_same_axes(other)
            timestamps = self.timestamps | other.timestamps
            self.ods = {
                timestamp: self._matrix_or_zero(timestamp) - other._matrix_or_zero(timestamp)
                for timestamp in timestamps
            }
            self.timestamps = timestamps
        elif isinstance(other, (int, float)):
            for matrix in self.ods.values():
                matrix -= other
        else:
            raise TypeError("Unsupported operand type for subtraction.")
        return self

    def __mul__(self, other: int | float | MatrixODT) -> MatrixODT:
        result = self.copy()
        result *= other
        return result

    def __imul__(self, other: int | float | MatrixODT) -> MatrixODT:
        if isinstance(other, MatrixODT):
            self._ensure_same_axes(other)
            timestamps = self.timestamps | other.timestamps
            self.ods = {
                timestamp: self._matrix_or_zero(timestamp) * other._matrix_or_zero(timestamp)
                for timestamp in timestamps
            }
            self.timestamps = timestamps
        elif isinstance(other, (int, float)):
            for matrix in self.ods.values():
                matrix *= other
        else:
            raise TypeError("Unsupported operand type for multiplication.")
        return self

    def __truediv__(self, other: int | float | MatrixODT) -> MatrixODT:
        result = self.copy()
        result /= other
        return result

    def __itruediv__(self, other: int | float | MatrixODT) -> MatrixODT:
        if isinstance(other, MatrixODT):
            self._ensure_same_axes(other)
            timestamps = self.timestamps | other.timestamps
            self.ods = {
                timestamp: self._matrix_or_zero(timestamp) / other._matrix_or_zero(timestamp)
                for timestamp in timestamps
            }
            self.timestamps = timestamps
        elif isinstance(other, (int, float)):
            for matrix in self.ods.values():
                matrix /= other
        else:
            raise TypeError("Unsupported operand type for division.")
        return self

    def nan_to_num(
        self,
        copy: bool = True,
        nan: float = 0.0,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> None:
        """Replace NaN and infinite values in every timestamp matrix."""
        for matrix in self.ods.values():
            matrix.nan_to_num(copy=copy, nan=nan, posinf=posinf, neginf=neginf)

    @staticmethod
    def read_df(
        rows: LabelsInput,
        cols: LabelsInput,
        timestamps: (
            list[Timestamp] | set[Timestamp] | tuple[Timestamp, ...] | pd.DataFrame | None
        ) = None,
        df: pd.DataFrame | None = None,
        o_field: str = "o",
        d_field: str = "d",
        timestamp_field: str = "timestamp",
        value_field: str = "value",
    ) -> MatrixODT:
        """Create a time-indexed matrix from a long-form DataFrame.

        For backwards compatibility, the third positional argument may be the
        DataFrame itself. When ``timestamps`` is omitted, it is inferred from the
        DataFrame in first-seen order.
        """
        if isinstance(timestamps, pd.DataFrame) and df is None:
            df = timestamps
            timestamps = None
        if df is None:
            raise TypeError("df is required.")

        frame = df[[o_field, d_field, timestamp_field, value_field]].rename(
            columns={
                o_field: "o",
                d_field: "d",
                timestamp_field: "timestamp",
                value_field: "value",
            }
        )
        if timestamps is None:
            timestamps = list(pd.unique(frame["timestamp"]))

        ods: dict[Timestamp, MatrixOD] = {}
        for timestamp, group in frame.groupby("timestamp", sort=False):
            ods[timestamp] = MatrixOD.read_df(
                rows=rows,
                cols=cols,
                df=group,
                o_field="o",
                d_field="d",
                value_field="value",
            )
        return MatrixODT(rows, cols, timestamps=timestamps, init=ods)

    @staticmethod
    def read_csv(
        rows: LabelsInput,
        cols: LabelsInput,
        file: str | PathLike[str],
        timestamps: list[Timestamp] | set[Timestamp] | tuple[Timestamp, ...] | None = None,
        o_field: str = "o",
        d_field: str = "d",
        timestamp_field: str = "timestamp",
        value_field: str = "value",
    ) -> MatrixODT:
        """Create a time-indexed matrix from a long-form CSV file."""
        df = pd.read_csv(file, usecols=[o_field, d_field, value_field, timestamp_field])
        return MatrixODT.read_df(
            rows=rows,
            cols=cols,
            timestamps=timestamps,
            df=df,
            o_field=o_field,
            d_field=d_field,
            timestamp_field=timestamp_field,
            value_field=value_field,
        )

    def write_df(
        self,
        o_field: str = "o",
        d_field: str = "d",
        timestamp_field: str = "timestamp",
        value_field: str = "value",
    ) -> pd.DataFrame:
        """Return the data as a sorted long-form DataFrame."""
        data: list[dict[str, Any]] = []
        for timestamp, od_matrix in self.ods.items():
            for origin, origin_index in od_matrix.rows.items():
                for destination, destination_index in od_matrix.cols.items():
                    data.append(
                        {
                            timestamp_field: timestamp,
                            o_field: origin,
                            d_field: destination,
                            value_field: od_matrix.mat[origin_index, destination_index],
                        }
                    )
        return pd.DataFrame(data).sort_values([timestamp_field, o_field, d_field])

    def write_csv(
        self,
        file: str | PathLike[str],
        o_field: str = "o",
        d_field: str = "d",
        timestamp_field: str = "timestamp",
        value_field: str = "value",
    ) -> None:
        """Write the data as a sorted long-form CSV file."""
        df = self.write_df(
            o_field=o_field,
            d_field=d_field,
            timestamp_field=timestamp_field,
            value_field=value_field,
        )
        df.to_csv(file, index=False)
