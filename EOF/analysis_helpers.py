"""Analysis helper utilities used by EOF/EOF_analysis.ipynb

Contains a vectorized implementation of Pearson correlation maps with an option
for chunked dask-backed processing, Benjamini-Hochberg FDR correction, and
plotting helpers to overlay significance contours.
"""

from typing import Tuple
import warnings

import numpy as np
import xarray as xr
from scipy.stats import t as t_dist


def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (one-dimensional array of p-values).

    Returns boolean mask of which p-values are significant at level alpha.
    """
    p = np.ascontiguousarray(pvals).ravel()
    m = p.size
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    compare = sorted_p <= thresholds
    if not np.any(compare):
        return np.zeros_like(p, dtype=bool)
    # largest k that satisfies
    k = np.max(np.where(compare)[0])
    cutoff = sorted_p[k]
    sig = p <= cutoff
    return sig.reshape(pvals.shape)


def _pearson_block(data_block: np.ndarray, index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pearson r and p for a data block (time, points) against index (time,).

    This is fully vectorized using numpy and handles NaNs by ignoring them.
    Returns arrays (r_vals, p_vals) with length = points.
    """
    # shapes
    T, P = data_block.shape
    if T != index.shape[0]:
        raise ValueError("time dimension mismatch between data block and index")

    # Masks of valid values
    valid = np.isfinite(data_block) & np.isfinite(index[:, None])
    n = valid.sum(axis=0)  # valid sample counts per point

    # Pre-allocate
    r = np.full(P, np.nan, dtype=float)
    p = np.full(P, np.nan, dtype=float)

    if P == 0:
        return r, p

    # Means (only over valid entries)
    # Avoid warnings for all-nan slices
    with np.errstate(invalid='ignore'):
        mean_x = np.where(n > 0, np.nansum(np.where(valid, data_block, 0.0), axis=0) / n, np.nan)
        mean_y = np.nansum(np.where(np.isfinite(index), index, 0.0)) / np.count_nonzero(np.isfinite(index))

    # Compute numerator and denominators using masked sums
    x_centered = np.where(valid, data_block - mean_x[None, :], 0.0)
    y_centered = np.where(valid, (index - mean_y)[:, None], 0.0)

    # Sum of cross-products and sums of squares
    ss_xy = np.sum(x_centered * y_centered, axis=0)
    ss_xx = np.sum(x_centered * x_centered, axis=0)
    ss_yy = np.sum(y_centered * y_centered, axis=0)

    # Use sample standard deviations (denominator uses (n-1))
    denom = ss_xx * ss_yy

    # Valid where n >= 3 and denom > 0
    ok = (n >= 3) & (denom > 0)
    if np.any(ok):
        r_ok = ss_xy[ok] / np.sqrt(denom[ok])
        # Clip to [-1,1] numerically
        r_ok = np.clip(r_ok, -1.0, 1.0)
        r[ok] = r_ok
        # t-statistic
        df = n[ok] - 2
        t_stat = r_ok * np.sqrt(df / (1.0 - r_ok ** 2))
        # two-sided p-value using Student's t survival function
        p[ok] = 2.0 * t_dist.sf(np.abs(t_stat), df)

    return r, p


def pearson_map(field: xr.DataArray, index: xr.DataArray, use_dask: bool = False, block_size: int = 20000) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute Pearson correlation map between a 3D field and a 1D index.

    Parameters
    ----------
    field : xr.DataArray
        Expected dims: (time, j, i) or (time, points) (time first).
    index : xr.DataArray
        1D time series with same time coordinate as `field` (or will be aligned).
    use_dask : bool
        If True and `field` is backed by dask, compute blockwise to limit memory.
        If False, computation is done in-memory with numpy (vectorized).
    block_size : int
        Number of spatial points to process per block when using chunking.

    Returns
    -------
    r_da, p_da : xr.DataArray
        Pearson r and p-value arrays with same spatial dimensions as `field` (unstacked if applicable).
    """
    # Align time coordinates
    field, index = xr.align(field, index, join='inner')
    if 'time' not in field.dims:
        raise ValueError("`field` must have a 'time' dimension")

    # Convert to stacked points if necessary
    if 'points' not in field.dims:
        spatial_dims = [d for d in field.dims if d != 'time']
        if len(spatial_dims) == 0:
            raise ValueError("No spatial dims found in `field`")
        field_st = field.stack(points=spatial_dims)
    else:
        field_st = field

    # Reorder to (time, points)
    field_st = field_st.transpose('time', 'points')

    # Prepare result arrays
    n_points = field_st.sizes['points']
    r_vals = np.full(n_points, np.nan, dtype=float)
    p_vals = np.full(n_points, np.nan, dtype=float)

    # Use dask-chunked processing by iterating blocks (safe memory usage)
    # If the underlying data is not dask or use_dask is False, we'll still process in blocks
    points = np.arange(n_points)
    idx_vals = index.values

    try:
        import dask.array as da
        is_dask = hasattr(field_st.data, 'chunks')
    except Exception:
        da = None
        is_dask = False

    if use_dask and is_dask:
        # iterate over blocks of points, compute each block with .compute()
        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            block = field_st.isel(points=slice(start, end)).data
            # .compute() will load the block as numpy
            block_np = np.asarray(block.compute())
            r_block, p_block = _pearson_block(block_np, idx_vals)
            r_vals[start:end] = r_block
            p_vals[start:end] = p_block
    else:
        # Non-dask or user chose not to use dask: process in blocks to bound memory
        # Try to get a numpy array view; if underlying is dask, compute whole array (may be large)
        try:
            arr = np.asarray(field_st.data)
        except Exception:
            # fallback to computing into memory
            arr = np.asarray(field_st.compute().data)

        # Process in blocks to avoid huge temporaries
        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            block_np = arr[:, start:end]
            r_block, p_block = _pearson_block(block_np, idx_vals)
            r_vals[start:end] = r_block
            p_vals[start:end] = p_block

    # Build DataArrays and unstack to original spatial dims
    r_da = xr.DataArray(r_vals, coords={'points': field_st['points']}, dims=['points'])
    p_da = xr.DataArray(p_vals, coords={'points': field_st['points']}, dims=['points'])
    try:
        r_da = r_da.unstack('points')
        p_da = p_da.unstack('points')
    except Exception:
        # if unstack fails, keep stacked dims
        pass

    # attach latitude/longitude if present on original field
    return r_da, p_da


def plot_map_with_significance(r_da: xr.DataArray, p_da: xr.DataArray, title: str = '', alpha: float = 0.05, method: str = 'pointwise', ax=None, cmap: str = 'RdBu_r', vmin: float = -0.6, vmax: float = 0.6, transform=None, **pcolormesh_kwargs):
    """Plot correlation map and overlay significance contours.

    method: 'pointwise' (simple p<alpha) or 'fdr' (Benjamini-Hochberg). Returns the matplotlib axes used.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    if ax is None:
        fig = plt.figure(figsize=(12, 4))
        ax = plt.axes(projection=transform or ccrs.PlateCarree())
    ax.coastlines()

    if method == 'fdr':
        # flatten p-values and apply BH
        p_flat = p_da.values.ravel()
        sig_mask_flat = fdr_bh(p_flat, alpha=alpha)
        sig_mask = sig_mask_flat.reshape(p_da.shape)
    else:
        sig_mask = (p_da < alpha).values

    # masked correlation for plotting
    data_plot = r_da.where(sig_mask)

    lon = None
    lat = None
    # try common coord names (flexible)
    for name in ('longitude', 'lon', 'long'):
        if name in r_da.coords:
            lon = r_da.coords[name]
            break
    for name in ('latitude', 'lat'):
        if name in r_da.coords:
            lat = r_da.coords[name]
            break

    # Convert to numpy for plotting, but be robust to 1D/2D coords or missing coords
    try:
        if lon is not None and lat is not None:
            lon_arr = np.asarray(lon)
            lat_arr = np.asarray(lat)
            # If both 2D, use them directly
            if lon_arr.ndim == 2 and lat_arr.ndim == 2:
                pcm = ax.pcolormesh(lon_arr, lat_arr, data_plot, transform=transform or ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', **pcolormesh_kwargs)
            # If both 1D, pcolormesh accepts (x, y, Z)
            elif lon_arr.ndim == 1 and lat_arr.ndim == 1:
                pcm = ax.pcolormesh(lon_arr, lat_arr, data_plot, transform=transform or ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', **pcolormesh_kwargs)
            else:
                # Mixed or unexpected dims -> fall back to imshow
                warnings.warn('Longitude/latitude have unexpected shapes; falling back to imshow for plotting.')
                pcm = ax.imshow(data_plot.values, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        else:
            # No coordinate information: use imshow with array indices
            warnings.warn('No latitude/longitude coords found; plotting with image indices.')
            pcm = ax.imshow(data_plot.values, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')

        plt.colorbar(pcm, ax=ax, orientation='vertical', label='Pearson r')
        ax.set_title(title)

        # overlay contour lines showing significance boundary
        try:
            import numpy as _np
            sig_numeric = _np.where(sig_mask, 1.0, 0.0)
            # If lon/lat are usable and 2D, draw contour in lon/lat space
            if lon is not None and lat is not None and lon_arr.ndim == 2 and lat_arr.ndim == 2:
                cs = ax.contour(lon_arr, lat_arr, sig_numeric, levels=[0.5], colors='k', linewidths=0.8, transform=transform or ccrs.PlateCarree())
                # don't label small contours; keep labels optional
            else:
                # contour in index space
                y_idx = np.arange(sig_numeric.shape[0])
                x_idx = np.arange(sig_numeric.shape[1])
                X, Y = np.meshgrid(x_idx, y_idx)
                cs = ax.contour(X, Y, sig_numeric, levels=[0.5], colors='k', linewidths=0.8)
        except Exception:
            warnings.warn('Could not overlay significance contours (coord mismatch or plotting error).')

    except Exception as e:
        warnings.warn(f'Plotting failed with error: {e}; falling back to imshow.')
        pcm = ax.imshow(data_plot.values, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        plt.colorbar(pcm, ax=ax, orientation='vertical', label='Pearson r')
        ax.set_title(title)

    return ax
