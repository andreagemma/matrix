# GA Matrix

[![CI](https://github.com/andreagemma/matrix/actions/workflows/ci.yml/badge.svg)](https://github.com/andreagemma/matrix/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ga-matrix.svg)](https://pypi.org/project/ga-matrix/)
[![Python](https://img.shields.io/pypi/pyversions/ga-matrix.svg)](https://pypi.org/project/ga-matrix/)

GA Matrix provides small NumPy-backed matrix containers with label-based access,
long-form pandas import/export helpers, and optional timestamp support for
origin-destination data.

The PyPI distribution is named `ga-matrix`; the import package is named `matrix`.

## Installation

```bash
python -m pip install ga-matrix
```

Development and test tools are available as extras:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[dev]"
```

## Quick Start

```python
from matrix import MatrixOD

rows = ["A", "B"]
cols = ["X", "Y"]

od = MatrixOD(rows, cols, init={"A": {"X": 10}, "B": {"Y": 5}})
od["A", "Y"] = 3

assert od["A", "X"] == 10
assert od.sum() == 18
```

## MatrixOD

`MatrixOD` stores a 2D origin-destination matrix. Rows and columns can be passed
as label sequences or existing label-to-position mappings.

```python
import pandas as pd

from matrix import MatrixOD

df = pd.DataFrame(
    [
        {"origin": "A", "destination": "X", "trips": 10},
        {"origin": "B", "destination": "Y", "trips": 5},
    ]
)

od = MatrixOD.read_df(
    rows=["A", "B"],
    cols=["X", "Y"],
    df=df,
    o_field="origin",
    d_field="destination",
    value_field="trips",
)

roundtrip = od.write_df(o_field="origin", d_field="destination", value_field="trips")
```

Supported operations are element-wise addition, subtraction, multiplication, and
division with either a scalar or another matrix with the same labels.

```python
scaled = od * 1.2
delta = scaled - od
col_totals = od.sum(axis=0)
row_totals = od.sum(axis=1)
```

## MatrixODT

`MatrixODT` stores one `MatrixOD` per timestamp.

```python
from matrix import MatrixODT

odt = MatrixODT(
    rows=["A", "B"],
    cols=["X", "Y"],
    timestamps=[0, 1],
    init={
        0: {"A": {"X": 10}},
        1: {"B": {"Y": 5}},
    },
)

assert odt["A", "X", 0] == 10
assert odt["A", "X", 99] == 0
assert odt.sum(axis=2)["A", "X"] == 10
```

`MatrixODT.read_df()` accepts a long-form DataFrame with origin, destination,
timestamp, and value columns. If `timestamps` is omitted, timestamp labels are
inferred from the DataFrame in first-seen order.

## LabeledMatrix

`LabeledMatrix` is a more generic 2D labeled array with `.loc` and `.iloc`
indexers:

```python
from matrix import LabeledMatrix

table = LabeledMatrix(
    [[1, 2], [3, 4]],
    row_index=["a", "b"],
    col_index=["x", "y"],
)

assert table.loc["b", "y"] == 4
```

## API Summary

- `MatrixOD(rows, cols, init=None, copy=False, mode=None)`
- `MatrixOD.read_df(rows, cols, df, o_field="o", d_field="d", value_field="value")`
- `MatrixOD.read_csv(rows, cols, file, ...)`
- `MatrixOD.write_df(...)`
- `MatrixOD.write_csv(file, ...)`
- `MatrixODT(rows, cols, timestamps, init=None, copy=False, mode=None)`
- `MatrixODT.read_df(rows, cols, timestamps=None, df=None, ...)`
- `MatrixODT.read_csv(rows, cols, file, timestamps=None, ...)`
- `MatrixODT.write_df(...)`
- `MatrixODT.write_csv(file, ...)`
- `LabeledMatrix(data, row_index=..., col_index=..., dtype=None, copy=False)`

## Development

GA Matrix supports Python 3.10 and newer.

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src
python -m pytest --cov=matrix --cov-report=term-missing
ruff check .
mypy
python -m pip check
python -m build
python -m twine check dist/*
```

## Releases

`src/matrix/_version.py` is the only version source. To publish a release:

1. Update `__version__` in `_version.py` and commit the release changes.
2. Push `main` and wait for CI to pass.
3. Configure the PyPI Trusted Publisher with project `ga-matrix`, owner
   `andreagemma`, repository `matrix`, workflow `release.yml`, and environment
   `pypi`.
4. Run the **Create release** GitHub Actions workflow. With no override it creates
   the `v<version>` tag, creates release notes, and dispatches the build and PyPI
   publication workflow.

PyPI versions are immutable. Increment `_version.py` before publishing different
content.

## License

GA Matrix is distributed under the MIT License. See [LICENSE](LICENSE).
