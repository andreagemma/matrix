# Changelog

All notable changes to this project will be documented in this file.

## 0.1.3 - Unreleased

- Added the public `MatrixODTSparse` time-indexed sparse matrix API and its
  DataFrame/CSV import and export support.
- Added timestamp accessors and shape metadata to `MatrixODT`.
- Preserved immutable label mappings and aligned sparse arithmetic, zero
  handling, shape reporting, and numerical cleanup with the dense API.
- Added third-party license inventory and packaged license texts.
- Bumped package/build version from `0.1.2` to `0.1.3`.

## 0.1.2

- Removed the `LabeledMatrix` class and its public API.
- Bumped package/build version from `0.1.1` to `0.1.2`.

## 0.1.1 - 2026-09-08

- Aligned GitHub Actions workflows with the configreader flow and analogous file names:
	- `all.yml`
	- `ci.yml`
	- `fast_ci.yml`
	- `quality.yml`
	- `create-release.yml`
	- `create-release-whl.yml`
	- `release.yml`
- Bumped package/build version from `0.1.0` to `0.1.1`.

## 0.1.0 - 2026-09-01

- Prepared the package for GitHub and PyPI publication as `ga-matrix`.
- Added README, MIT license, gitignore, package version source, and typed package marker.
- Corrected packaging metadata, project URLs, coverage source, and GitHub Actions imports.
- Reworked tests around `pytest` and removed heavyweight performance checks from unit tests.
- Fixed in-place subtraction and division behavior in matrix operations.
