from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matrix import MatrixOD, MatrixODT


def make_matrix_odt() -> MatrixODT:
    return MatrixODT(
        rows=["A", "B", "C"],
        cols=["X", "Y", "Z"],
        timestamps=[0, 1],
        init={
            0: {
                "A": {"X": 5, "Y": 3, "Z": 0},
                "B": {"X": 0, "Y": 0, "Z": 7},
                "C": {"X": 0, "Y": 0, "Z": 0},
            },
            1: {
                "A": {"X": 10, "Y": 20, "Z": 30},
                "B": {"X": 40, "Y": 50, "Z": 60},
                "C": {"X": 70, "Y": 80, "Z": 90},
            },
        },
    )


def test_initialization_label_access_and_missing_timestamp() -> None:
    matrix = make_matrix_odt()

    assert matrix["A", "X", 0] == 5
    assert matrix["B", "Z", 0] == 7
    assert matrix["A", "X", 1] == 10
    assert matrix["C", "Z", 1] == 90
    assert matrix["A", "X", 99] == 0
    assert matrix[99].sum() == 0


def test_setitem_updates_existing_and_new_timestamps() -> None:
    matrix = make_matrix_odt()

    matrix["A", "Y", 0] = 15
    matrix["B", "X", 2] = 9

    assert matrix["A", "Y", 0] == 15
    assert matrix["B", "X", 2] == 9
    assert 2 in matrix.timestamps


def test_setitem_replaces_timestamp_matrix() -> None:
    matrix = make_matrix_odt()
    replacement = MatrixOD(matrix.rows, matrix.cols, init=3)

    matrix[3] = replacement

    assert matrix["A", "X", 3] == 3
    assert matrix[3] is replacement


def test_getitem_key_error_for_missing_origin_or_destination() -> None:
    matrix = make_matrix_odt()

    with pytest.raises(KeyError):
        _ = matrix["D", "X", 0]


def test_copy_is_deep_by_default() -> None:
    matrix = make_matrix_odt()
    matrix_copy = matrix.copy(copy_data=True)

    matrix_copy["A", "X", 0] = 100

    assert matrix["A", "X", 0] == 5
    assert matrix_copy["A", "X", 0] == 100


def test_sum_axes() -> None:
    matrix = make_matrix_odt()

    assert matrix.sum() == 465

    col_totals = matrix.sum(axis=0)
    assert isinstance(col_totals, MatrixODT)
    assert col_totals["sum", "X", 0] == 5
    assert col_totals["sum", "Z", 1] == 180

    row_totals = matrix.sum(axis=1)
    assert isinstance(row_totals, MatrixODT)
    assert row_totals["A", "sum", 0] == 8
    assert row_totals["C", "sum", 1] == 240

    timestamp_totals = matrix.sum(axis=2)
    assert isinstance(timestamp_totals, MatrixOD)
    assert timestamp_totals["A", "X"] == 15
    assert timestamp_totals["C", "Z"] == 90

    with pytest.raises(ValueError, match="Axis"):
        matrix.sum(axis=4)


def test_scalar_arithmetic_uses_expected_operators() -> None:
    matrix = make_matrix_odt()

    matrix -= 1
    assert matrix["A", "X", 0] == 4

    matrix /= 2
    assert matrix["A", "X", 0] == 2

    doubled = matrix * 2
    assert doubled["A", "X", 0] == 4


def test_matrix_arithmetic_uses_timestamp_union() -> None:
    left = MatrixODT(["A"], ["X"], timestamps=[0], init={0: {"A": {"X": 4}}})
    right = MatrixODT(["A"], ["X"], timestamps=[1], init={1: {"A": {"X": 2}}})

    added = left + right
    subtracted = left - right
    multiplied = left * right

    assert added["A", "X", 0] == 4
    assert added["A", "X", 1] == 2
    assert subtracted["A", "X", 0] == 4
    assert subtracted["A", "X", 1] == -2
    assert multiplied["A", "X", 0] == 0
    assert multiplied["A", "X", 1] == 0


def test_read_df_infers_timestamps_and_accepts_old_positional_form() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": 0, "o": "A", "d": "X", "value": 5},
            {"timestamp": 1, "o": "B", "d": "Z", "value": 60},
            {"timestamp": 1, "o": "C", "d": "Y", "value": 80},
        ]
    )

    matrix = MatrixODT.read_df(["A", "B", "C"], ["X", "Y", "Z"], frame)

    assert matrix.timestamps == {0, 1}
    assert matrix["A", "X", 0] == 5
    assert matrix["B", "Z", 1] == 60
    assert matrix["C", "Y", 1] == 80


def test_read_df_accepts_explicit_timestamps() -> None:
    frame = pd.DataFrame([{"timestamp": 0, "o": "A", "d": "X", "value": 5}])

    matrix = MatrixODT.read_df(["A"], ["X"], timestamps=[0, 1], df=frame)

    assert matrix.timestamps == {0, 1}
    assert matrix["A", "X", 0] == 5
    assert matrix["A", "X", 1] == 0


def test_csv_and_dataframe_roundtrip(tmp_path) -> None:
    matrix = make_matrix_odt()
    path = tmp_path / "matrix_odt.csv"

    matrix.write_csv(path)
    loaded = MatrixODT.read_csv(["A", "B", "C"], ["X", "Y", "Z"], path)

    np.testing.assert_array_equal(loaded[0].mat, matrix[0].mat)
    np.testing.assert_array_equal(loaded[1].mat, matrix[1].mat)
    assert len(matrix.write_df()) == 18
