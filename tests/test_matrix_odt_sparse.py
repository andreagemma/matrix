from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from matrix import MatrixODTSparse


def make_sparse() -> MatrixODTSparse:
    """Create a small sparse time-indexed matrix for tests."""
    return MatrixODTSparse(
        rows=["A", "B"],
        cols=["X", "Y"],
        timestamps=[0, 1],
        init={("A", "X", 0): 5, ("B", "Y", 1): 3},
    )


def test_initialization_access_and_shape() -> None:
    matrix = make_sparse()

    assert matrix["A", "X", 0] == 5
    assert matrix["A", "Y", 0] == 0
    assert matrix["B", "Y", 1] == 3
    assert matrix.shape == (2, 2, 2)
    assert len(list(matrix.get_all_od())) == 2
    assert matrix.sum() == 8

    at_zero = matrix[0]
    assert isinstance(at_zero, MatrixODTSparse)
    assert at_zero.timestamps == {0}
    assert at_zero["A", "X", 0] == 5


def test_integer_scalar_initialization_and_zero_elision() -> None:
    matrix = MatrixODTSparse(["A", "B"], ["X"], [0], init=2)

    assert matrix.ods == {("A", "X", 0): 2.0, ("B", "X", 0): 2.0}
    matrix["A", "X", 0] = 0
    assert ("A", "X", 0) not in matrix.ods
    assert matrix.timestamps == {0}
    assert MatrixODTSparse(["A"], ["X"], [0], init=0).ods == {}


def test_validates_labels_and_tracks_initialized_timestamps() -> None:
    matrix = MatrixODTSparse(["A"], ["X"], [0], init={("A", "X", 2): 4})

    assert matrix.timestamps == {0, 2}
    with pytest.raises(KeyError, match="Row label"):
        _ = matrix["B", "X", 0]
    with pytest.raises(KeyError, match="Column label"):
        matrix["A", "Y", 3] = 1
    assert matrix.timestamps == {0, 2}
    with pytest.raises(ValueError, match="same row and column labels"):
        _ = matrix + MatrixODTSparse(["B"], ["X"], [0])


def test_copy_can_be_deep_or_shared() -> None:
    matrix = make_sparse()
    deep_copy = matrix.copy()
    shared_copy = matrix.copy(copy_data=False)

    deep_copy["A", "X", 0] = 9
    assert matrix["A", "X", 0] == 5

    shared_copy["A", "X", 0] = 9
    assert matrix["A", "X", 0] == 9


def test_arithmetic_matches_zero_filled_sparse_semantics() -> None:
    left = MatrixODTSparse(
        ["A", "B"],
        ["X", "Y"],
        [0],
        init={("A", "X", 0): 6, ("B", "Y", 0): 4},
    )
    right = MatrixODTSparse(
        ["A", "B"],
        ["X", "Y"],
        [0, 1],
        init={("A", "X", 0): 2, ("B", "Y", 1): 3},
    )

    added = left + right
    subtracted = left - right
    multiplied = left * right

    assert added["A", "X", 0] == 8
    assert added["B", "Y", 0] == 4
    assert added["B", "Y", 1] == 3
    assert subtracted["A", "X", 0] == 4
    assert subtracted["B", "Y", 1] == -3
    assert multiplied["A", "X", 0] == 12
    assert multiplied["B", "Y", 0] == 0
    assert multiplied.timestamps == {0, 1}
    assert multiplied.ods == {("A", "X", 0): 12}

    assert (2 + left)["A", "Y", 0] == 2
    assert (2 - left)["A", "X", 0] == -4
    assert (2 * left)["A", "X", 0] == 12
    assert math.isinf((2 / left)["A", "Y", 0])

    left *= 0
    assert left.ods == {}


def test_division_preserves_ieee_zero_behavior() -> None:
    matrix = MatrixODTSparse(["A", "B"], ["X"], [0], init={("A", "X", 0): 2})
    matrix /= 0

    assert math.isinf(matrix["A", "X", 0])
    assert math.isnan(matrix["B", "X", 0])

    divided = make_sparse() / MatrixODTSparse(["A", "B"], ["X", "Y"], [0, 1])
    assert math.isinf(divided["A", "X", 0])
    assert math.isnan(divided["B", "X", 0])


def test_nan_to_num_and_sparse_dataframe_roundtrip(tmp_path: Path) -> None:
    matrix = MatrixODTSparse(
        ["A", "B"],
        ["X"],
        [0],
        init={
            ("A", "X", 0): float("nan"),
            ("B", "X", 0): float("inf"),
        },
    )
    matrix.nan_to_num()

    assert matrix["A", "X", 0] == 0
    assert matrix["B", "X", 0] == sys.float_info.max
    assert ("A", "X", 0) not in matrix.ods

    frame = pd.DataFrame(
        [
            {"time": 0, "from": "A", "to": "X", "flow": 5},
            {"time": 1, "from": "B", "to": "Y", "flow": 3},
        ]
    )
    loaded = MatrixODTSparse.read_df(
        ["A", "B"],
        ["X", "Y"],
        df=frame,
        o_field="from",
        d_field="to",
        timestamp_field="time",
        value_field="flow",
    )
    roundtrip = MatrixODTSparse.read_df(["A", "B"], ["X", "Y"], df=loaded.write_df())

    assert loaded.timestamps == {0, 1}
    assert roundtrip["A", "X", 0] == 5
    assert roundtrip["B", "Y", 1] == 3
    assert list(loaded.write_df(reset_index=False).index.names) == ["o", "d", "timestamp"]

    path = tmp_path / "sparse.csv"
    loaded.write_csv(path)
    csv_loaded = MatrixODTSparse.read_csv(["A", "B"], ["X", "Y"], path)
    assert csv_loaded["A", "X", 0] == 5
    assert csv_loaded["B", "Y", 1] == 3

    empty = MatrixODTSparse(["A"], ["X"], [0]).write_df()
    assert list(empty.columns) == ["o", "d", "timestamp", "value"]
    assert empty.empty
