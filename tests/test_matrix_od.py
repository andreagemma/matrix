from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matrix import MatrixOD


def make_matrix() -> MatrixOD:
    return MatrixOD(
        rows=["A", "B", "C"],
        cols=["X", "Y", "Z"],
        init=[[5, 3, 0], [0, 0, 7], [0, 0, 0]],
    )


def test_initialization_and_label_access() -> None:
    matrix = make_matrix()

    assert matrix["A", "X"] == 5
    assert matrix["A", "Y"] == 3
    assert matrix["B", "Z"] == 7
    assert matrix["C", "X"] == 0


def test_setitem_and_missing_labels() -> None:
    matrix = make_matrix()
    matrix["C", "Y"] = 10

    assert matrix["C", "Y"] == 10
    with pytest.raises(KeyError):
        _ = matrix["D", "X"]
    with pytest.raises(KeyError):
        matrix["A", "missing"] = 1


def test_validates_labels_and_shape() -> None:
    with pytest.raises(ValueError, match="Duplicate label"):
        MatrixOD(["A", "A"], ["X"])

    with pytest.raises(ValueError, match="shape"):
        MatrixOD(["A", "B"], ["X", "Y"], init=[[1, 2, 3]])


def test_copy_can_be_deep_or_shared() -> None:
    matrix = make_matrix()
    deep_copy = matrix.copy(copy_data=True)
    shared_copy = matrix.copy(copy_data=False)

    deep_copy["A", "X"] = 99
    assert matrix["A", "X"] == 5

    shared_copy["A", "X"] = 42
    assert matrix["A", "X"] == 42


def test_arithmetic_with_scalars_and_matrices() -> None:
    matrix = make_matrix()
    other = MatrixOD(matrix.rows, matrix.cols, init=2)

    np.testing.assert_array_equal((matrix + 1).mat, matrix.mat + 1)
    np.testing.assert_array_equal((1 + matrix).mat, matrix.mat + 1)
    np.testing.assert_array_equal((matrix - 1).mat, matrix.mat - 1)
    np.testing.assert_array_equal((10 - matrix).mat, 10 - matrix.mat)
    np.testing.assert_array_equal((matrix * 2).mat, matrix.mat * 2)
    np.testing.assert_array_equal((2 * matrix).mat, matrix.mat * 2)
    np.testing.assert_array_equal((matrix / 2).mat, matrix.mat / 2)
    np.testing.assert_array_equal((matrix + other).mat, matrix.mat + other.mat)
    np.testing.assert_array_equal((matrix - other).mat, matrix.mat - other.mat)
    np.testing.assert_array_equal((matrix * other).mat, matrix.mat * other.mat)
    np.testing.assert_array_equal((matrix / other).mat, matrix.mat / other.mat)


def test_inplace_subtraction_and_division() -> None:
    matrix = make_matrix()
    other = MatrixOD(matrix.rows, matrix.cols, init=1)

    matrix -= other
    np.testing.assert_array_equal(matrix.mat, np.array([[4, 2, -1], [-1, -1, 6], [-1, -1, -1]]))

    matrix -= 1
    np.testing.assert_array_equal(matrix.mat, np.array([[3, 1, -2], [-2, -2, 5], [-2, -2, -2]]))

    matrix /= 2
    np.testing.assert_array_equal(
        matrix.mat,
        np.array([[1.5, 0.5, -1], [-1, -1, 2.5], [-1, -1, -1]]),
    )


def test_dimension_mismatch_raises() -> None:
    matrix = make_matrix()
    other = MatrixOD(["A", "B"], ["X", "Y"])

    with pytest.raises(ValueError, match="same row and column labels"):
        _ = matrix + other


def test_sum_axes_return_labeled_matrices() -> None:
    matrix = make_matrix()

    assert matrix.sum() == 15

    col_totals = matrix.sum(axis=0)
    assert isinstance(col_totals, MatrixOD)
    assert col_totals["sum", "X"] == 5
    assert col_totals["sum", "Y"] == 3
    assert col_totals["sum", "Z"] == 7

    row_totals = matrix.sum(axis=1)
    assert isinstance(row_totals, MatrixOD)
    assert row_totals["A", "sum"] == 8
    assert row_totals["B", "sum"] == 7
    assert row_totals["C", "sum"] == 0

    with pytest.raises(ValueError, match="Axis"):
        matrix.sum(axis=3)


def test_linear_algebra_and_nan_cleanup() -> None:
    matrix = MatrixOD(["A", "B"], ["A", "B"], init=[[4, 7], [2, 6]])

    assert matrix.transpose()["B", "A"] == 7
    np.testing.assert_allclose(matrix.inverse().mat, np.array([[0.6, -0.7], [-0.2, 0.4]]))

    matrix.set_diagonal([1, 2])
    np.testing.assert_array_equal(matrix.get_diagonal(), np.array([1, 2]))

    dirty = MatrixOD(["A"], ["B"], init=[[np.nan]])
    dirty.nan_to_num()
    assert dirty["A", "B"] == 0


def test_dataframe_and_csv_roundtrip(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"origin": "A", "destination": "X", "trips": 5},
            {"origin": "B", "destination": "Y", "trips": 7},
        ]
    )

    matrix = MatrixOD.read_df(
        rows=["A", "B"],
        cols=["X", "Y"],
        df=frame,
        o_field="origin",
        d_field="destination",
        value_field="trips",
    )

    assert matrix["A", "X"] == 5
    assert matrix["B", "Y"] == 7

    path = tmp_path / "matrix.csv"
    matrix.write_csv(path, o_field="origin", d_field="destination", value_field="trips")
    loaded = MatrixOD.read_csv(
        rows=["A", "B"],
        cols=["X", "Y"],
        file=path,
        o_field="origin",
        d_field="destination",
        value_field="trips",
    )

    np.testing.assert_array_equal(loaded.mat, matrix.mat)
