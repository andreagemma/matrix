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

## `LabeledMatrix`

```python
LabeledMatrix(data, row_index=..., col_index=..., dtype=None, copy=False)
```

Creates a generic labeled 2D NumPy array. Use direct indexing for scalar access,
`.loc` for label-based indexing, and `.iloc` for position-based indexing.

Main methods:

- `from_dict_of_dicts(data, dtype=None, row_order=None, col_order=None, fill_value=0)`
- `reindex(rows=None, cols=None, fill_value=0)`
- `rename(rows=None, cols=None)`
- `tolist()`
- `to_numpy(copy=False)`
- `to_pandas()`
