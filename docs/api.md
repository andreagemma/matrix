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
