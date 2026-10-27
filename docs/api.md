# API Reference

## `MatrixOD`

```python
MatrixOD(rows, cols, init=None, copy=False, mode=None)
```

Creates a NumPy-backed matrix addressed by row and column labels.

Main methods:

- `copy(copy_data=True)`
- `transpose()`
- `inverse()`
- `get_diagonal()`
- `set_diagonal(values)`
- `nan_to_num(copy=True, nan=0.0, posinf=None, neginf=None)`
- `sum(axis=None)`
- `read_df(rows, cols, df, o_field="o", d_field="d", value_field="value")`
- `read_csv(rows, cols, file, ...)`
- `write_df(...)`
- `write_csv(file, ...)`

Arithmetic with scalars or label-aligned `MatrixOD` instances is element-wise.

## `MatrixODT`

```python
MatrixODT(rows, cols, timestamps, init=None, copy=False, mode=None)
```

Creates a timestamp-indexed collection of `MatrixOD` objects. Scalar access uses
`matrix[origin, destination, timestamp]`; timestamp access uses `matrix[timestamp]`.

Main methods:

- `copy(copy_data=True)`
- `sum(axis=None)`
- `nan_to_num(copy=True, nan=0.0, posinf=None, neginf=None)`
- `read_df(rows, cols, timestamps=None, df=None, ...)`
- `read_csv(rows, cols, file, timestamps=None, ...)`
- `write_df(...)`
- `write_csv(file, ...)`

Arithmetic with scalars or label-aligned `MatrixODT` instances is element-wise.
When two `MatrixODT` objects contain different timestamps, missing timestamps are
treated as zero matrices.

`get_matrix_at_t(timestamp)` returns the matrix at a timestamp, or a zero matrix
when the timestamp is absent. `get_od(timestamp, origin, destination)` returns
one value and returns zero for a missing timestamp. `shape` is
`(n_timestamps, n_origins, n_destinations)` when timestamp matrices exist.

## `MatrixODTSparse`

```python
MatrixODTSparse(rows, cols, timestamps, init=None, copy=False, mode=None)
```

Stores nonzero values in a sparse mapping keyed by
`(origin, destination, timestamp)`. Unstored coordinates are zero, and setting
a coordinate to zero removes it from storage. A nonzero scalar initializer
materializes every coordinate; coordinate mappings are preferred for sparse
initialization.

Main methods:

- `copy(copy_data=True)`
- `get_od(origin, destination, timestamp)`
- `get_all_od()`
- `sum()`
- `nan_to_num(copy=True, nan=0.0, posinf=None, neginf=None)`
- `read_df(rows, cols, timestamps=None, df=None, ...)`
- `read_csv(rows, cols, file, timestamps=None, ...)`
- `write_df(...)`
- `write_csv(file, ...)`

Arithmetic supports scalars, coordinate mappings, and axis-compatible
`MatrixODTSparse` instances. Missing timestamps and coordinates are treated as
zero. Division by zero follows NumPy's IEEE NaN/infinity behavior.
