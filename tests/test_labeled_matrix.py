from __future__ import annotations

import numpy as np
import pytest

from matrix import LabeledMatrix


def test_labeled_matrix_direct_loc_and_iloc_access() -> None:
    matrix = LabeledMatrix(
        [[1, 2], [3, 4]],
        row_index=["a", "b"],
        col_index=["x", "y"],
    )

    assert matrix["a", "x"] == 1
    assert matrix.iloc[1, 1] == 4
    assert matrix.loc["b", "y"] == 4
    np.testing.assert_array_equal(matrix.loc["a", ["x", "y"]], np.array([1, 2]))

    matrix.loc["a", "x"] = 9
    assert matrix["a", "x"] == 9


def test_labeled_matrix_construction_validation() -> None:
    with pytest.raises(ValueError, match="2D"):
        LabeledMatrix([1, 2, 3], row_index=["a"], col_index=["x"])

    with pytest.raises(ValueError, match="unique"):
        LabeledMatrix([[1], [2]], row_index=["a", "a"], col_index=["x"])


def test_from_dict_reindex_rename_and_arithmetic() -> None:
    matrix = LabeledMatrix.from_dict_of_dicts({"a": {"x": 1}, "b": {"y": 4}})

    assert matrix["a", "x"] == 1
    assert matrix["a", "y"] == 0

    reindexed = matrix.reindex(rows=["b", "c"], cols=["y", "z"], fill_value=-1)
    assert reindexed["b", "y"] == 4
    assert reindexed["c", "z"] == -1

    renamed = matrix.rename(rows={"a": "alpha"}, cols={"x": "ex"})
    assert renamed["alpha", "ex"] == 1

    np.testing.assert_array_equal((matrix + 1).to_numpy(), matrix.to_numpy() + 1)


def test_labeled_matrix_requires_aligned_labels_for_matrix_arithmetic() -> None:
    left = LabeledMatrix([[1]], row_index=["a"], col_index=["x"])
    right = LabeledMatrix([[1]], row_index=["b"], col_index=["x"])

    with pytest.raises(ValueError, match="Labels must align"):
        _ = left + right
