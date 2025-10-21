Overview of `EOF/EOF_analysis.ipynb`

Summary
-------
This notebook performs individual and combined Empirical Orthogonal Function (EOF) analyses and correlation tests between ocean surface fields and the Atlantic Meridional Overturning Circulation (AMOC) index. The main variables are:
- SST (sea surface temperature) from `thetao_masked.nc`
- SSS (sea surface salinity) from `so_masked.nc`
- MLT (mixed layer thickness) from `mlotst_masked.nc`
- AMOC diagnostics (`msftyz` netCDF)

Primary outputs
---------------
- EOF spatial patterns (eofs) and principal components (pcs) for SST, SSS, and MLT
- Variance explained by leading EOFs
- Time series plots of leading PCs
- Pearson correlation maps between spatial fields (SST/SSS/MLT) and a single-point AMOC index
- Correlations between principal components and the AMOC index (table + scatter)
- A combined dataset (standardized & stacked) prepared for joint EOF analysis

High-level steps
----------------
1. Load required datasets using `xarray.open_dataset` and extract the variable of interest (e.g., `thetao`, `so`, `mlotst`).
2. Select surface/near-surface layer (for `thetao`/`so` use `isel(lev=0)` if 3D).
3. Compute time anomalies by subtracting the time mean for each field: `anom = var - var.mean(dim='time')`.
4. Compute cosine-latitude weights: `weights = sqrt(cos(lat_in_radians))` for area-weighting EOFs.
5. EOF analysis using `eofs.xarray.Eof` with a helper `compute_eofs(data, weights, n_modes=3)` that returns `eofs, pcs, variance`.
6. Plot EOF spatial patterns using cartopy (`pcolormesh`) and PCs as time series.
7. Load AMOC (`msftyz`), extract an index (e.g., Atlantic basin, 26N, 1000 m) and align times with PCs.
8. Compute Pearson correlations between PCs and AMOC (function `correlate_pcs_with_amoc`) and visualize the top correlations.
9. Compute spatial Pearson correlation maps between fields and the AMOC index using `pearson_map(field, index)`. This function handles both (time, j, i) arrays and stacked `points` dimensions.
10. Create a combined dataset by stacking spatial dims into `points`, standardizing each variable, and concatenating along a new `variable` dimension for joint EOF.

Key functions in the notebook
-----------------------------
- compute_eofs(data, weights, n_modes=3)
  - Uses `eofs.xarray.Eof` to compute EOF patterns, PCs, and explained variance.
  - Inputs: `data` (xarray.DataArray with dims (time, j, i) or similar), `weights` (1D lat weights), `n_modes`.
  - Outputs: `eofs` (spatial modes), `pcs` (time series), `variance` (fractional variance explained).

- plot_eofs_pcs(eofs, pcs, variance, varname)
  - Plots leading EOFs and their PCs in a 3x2 grid.

- correlate_pcs_with_amoc(pcs, amoc, varname)
  - Computes Pearson r and p-values between each PC and the AMOC index and returns a pandas DataFrame with results.

- pearson_map(field, index)
  - Computes Pearson r and p-value at each spatial point.
  - Handles both flattened `points` (stacked) and (time, j, i) fields.
  - Returns `r_da` and `p_da` as xarray.DataArray with coordinates for `latitude` and `longitude`.

Data shapes and contracts
------------------------
- SST/SSS:
  - Likely original shape: (time, lev, j, i). After selecting surface: (time, j, i).
- MLT:
  - Often already 2D: (time, j, i) or (time, points).
- EOF solver:
  - Expects (time, j, i) or DataArray that Eof can accept; weights must match lat dimension or be broadcastable.

Edge cases and failure modes
---------------------------
- Missing or differently named coordinate dims (e.g., `lat` vs `latitude`, `lon` vs `longitude`, or `j/i` vs `y/x`) can break indexing and plotting.
- NaNs: the `pearson_map` function checks finite values and requires at least 3 valid points to compute Pearson r; otherwise returns NaN.
- Time mismatch between datasets: notebook uses `.interp()` and `np.intersect1d` to align times, but care is needed if calendars or units differ.
- Memory: stacking large global fields into `points` before concatenation can use lots of memory; consider dask-backed xarray or streaming.
- Multiple testing: maps show pointwise p-values but do not correct for multiple comparisons (e.g., FDR or Bonferroni).

Performance notes & suggestions
------------------------------
- Speed up `pearson_map` by vectorizing with xarray.apply_ufunc or using numpy masked arrays. Replacing Python loops over grid points with vectorized operations will be much faster.
- Use dask arrays via xarray to handle large datasets without loading everything into memory.
- Use `xr.broadcast` and consistent coordinate names to avoid coordinate alignment issues.
- Save intermediate results (e.g., EOFs, PCs, correlation maps) to NetCDF files to avoid re-computation.
- Add command-line arguments or a small driver script to run the analysis reproducibly and with configurable paths.
- Add multiple-testing correction (e.g., Benjamini-Hochberg) when masking correlation maps based on p-values.

Reproducibility / environment
----------------------------
The notebook relies on these core packages (visible in code): `xarray`, `numpy`, `matplotlib`, `cartopy`, `eofs`, `scipy`, `pandas`.
Consider adding a `requirements.txt` or `environment.yaml` (there is an `environment.yaml` at repo root) that pins versions used for the analysis.

Where to look in the notebook
----------------------------
- Data loading and preprocessing: top cells (loading `thetao`, `so`, `mlotst`) and anomaly computation.
- EOF computation: cell with `compute_eofs` and calls for `eofs_sst`, `eofs_sss`, `eofs_mlt`.
- Correlation with AMOC and mapping: cells containing `pearson_map` and `plot_map`.
- Combined EOF prep: final cell stacking `points` and standardizing variables.

Next steps I can help with
-------------------------
- Turn the notebook into a reusable script or package with functions and a CLI.
- Vectorize and speed up `pearson_map` and add dask support.
- Add automatic saving of results and unit tests for small helper functions.
- Implement multiple-testing correction and annotate maps with significance contours.

