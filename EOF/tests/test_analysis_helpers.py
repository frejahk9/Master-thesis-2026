import numpy as np
import xarray as xr

from EOF.analysis_helpers import pearson_map, fdr_bh


def test_pearson_map_simple():
    # create synthetic data: time x j x i
    T = 100
    j, i = 2, 2
    time = np.arange(T)
    # index is a linear trend
    index = np.linspace(0, 1, T)

    # Field: one point correlated with index, others random noise
    rng = np.random.RandomState(0)
    field = np.zeros((T, j, i))

    # strongly correlated point at (0,0)
    field[:, 0, 0] = index + 0.1 * rng.normal(size=T)
    # weakly correlated point at (0,1)
    field[:, 0, 1] = 0.01 * rng.normal(size=T)
    # add NaNs for one point
    field[:10, 1, 0] = np.nan
    field[:, 1, 1] = rng.normal(size=T)

    da = xr.DataArray(field, dims=('time', 'j', 'i'), coords={'time': time, 'j': np.arange(j), 'i': np.arange(i)})
    idx_da = xr.DataArray(index, dims=('time',), coords={'time': time})

    r_da, p_da = pearson_map(da, idx_da, use_dask=False, block_size=2)

    # check shapes
    assert set(r_da.dims) >= {'j', 'i'}
    assert set(p_da.dims) >= {'j', 'i'}

    r00 = float(r_da.sel(j=0, i=0))
    p00 = float(p_da.sel(j=0, i=0))
    r01 = float(r_da.sel(j=0, i=1))
    p01 = float(p_da.sel(j=0, i=1))

    # strongly correlated: r high, p small
    assert r00 > 0.8
    assert p00 < 0.01

    # weakly correlated: r near zero, p large
    assert abs(r01) < 0.2
    assert p01 > 0.05


def test_fdr_bh():
    pvals = np.array([0.001, 0.02, 0.04, 0.5])
    sig = fdr_bh(pvals, alpha=0.05)
    # expected: first two significant under BH
    assert sig.dtype == bool
    assert sig.tolist() == [True, True, False, False]
