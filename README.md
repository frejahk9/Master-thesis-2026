# Simplified notebooks — hydrographic predictors of AMOC variability

Two lightweight notebooks that reproduce the core analyses of
*Low-dimensional hydrographic predictors of AMOC variability in climate models*
(Devilliers, Beauchamp, Klejnstrup, Christiansen, Karoff, Knudsen, 2026),
condensing the pipeline of `Final_figures/Paper_notebook.ipynb`.

| Notebook | Where it runs | What it does |
|---|---|---|
| `01_extract_and_save_data.ipynb` | Server (needs the CMIP6 `*_masked*.nc` files and `AMOC_{MODEL}.nc`) | Extracts the latitude–depth sections and saves them as compact NetCDF files |
| `02_EOF_AMOC_reconstruction.ipynb` | Anywhere (reads only `data/`) | EOFs, predictor ranking and selection, AMOC reconstruction, key figures |

Models: CESM2, EC-Earth3, IPSL-CM6A-LR, MPI-ESM1-2-LR (historical ensembles, 1850–2014).

## Quick start

```bash
# 1. On the server: extract and save the data (once)
jupyter nbconvert --to notebook --execute 01_extract_and_save_data.ipynb

# 2. Copy the data/ folder to your machine, then run the analysis
jupyter nbconvert --to notebook --execute 02_EOF_AMOC_reconstruction.ipynb
```

Dependencies: `numpy`, `pandas`, `xarray`, `netCDF4`, `scikit-learn`, `matplotlib`. The `eofs` package is **not** required.

## 1 · Data extraction (`01`)

For each model × variable (`thetao`, `so`) × longitude (10, 20, 30, 40, 60°W), the notebook keeps, for every grid row,
the nearest valid ocean point within ±10° of the target longitude. This is the same geometry as in the repository code.
Each member file is opened only once.

Output in `data/`:

```
{MODEL}_thetao_sections.nc   thetao_W10p0 ... (member, year, lat_W10p0, depth)  anomalies
                             thetao_W10p0_clim (member, lat_W10p0, depth)        climatology (≤ 1989)
                             dz (depth)
{MODEL}_so_sections.nc       same for salinity
{MODEL}_amoc.nc              AMOC_26N_* and AMOC_45N_* series on a `year` axis
```

Raw values can be recovered as `anomaly + clim`.

### Size reduction settings

| Setting | Default | Effect |
|---|---|---|
| `MODELS`, `VARS` | all | Run one model or variable at a time; existing files are skipped |
| `LAT_RES` | `1.0` | Rows averaged onto a regular latitude grid (`None` = native grid rows) |
| `DEPTH_MAX` | `5000` m | Maximum depth kept |
| `LSD` | T: 3, S: 4 | Significant decimals kept (zlib compression + quantisation) |
| `MAX_MEMBERS` | `None` | Cap on the number of ensemble members |

The last cell prints the size of each file.

## 2 · Analysis (`02`)

1. **EOFs**: for each section, the weights are √(cos φ · Δz). EOFs are fitted on the training period (1850–1989) with all members stacked. The whole period (1850–2014) is then projected onto them. The ensemble-mean PC is the mean of the member PCs. Results are cached in `results/{MODEL}_eofs_*.npz`.
2. **Ranking**: every (variable, longitude, EOF mode 1–10, lag `MIN_LAG`–20 yr) feature is ranked by its |correlation| with the AMOC over the training period.
3. **Selection**: an exhaustive search over the top `N_CANDIDATES` features finds the best subsets for k = 1, 2, 3. The model is ridge regression (α = 1, standardised inputs), and subsets are selected on training R².
4. **Evaluation**: out-of-sample R² and correlation are computed on 1990–2014.

Outputs in `results/`:

| File | Paper equivalent |
|---|---|
| `reconstruction_skill_AMOC45N.csv` | Table 1 / Extended Data Table 1 |
| `fig1_eof_pc_{MODEL}.png` | Fig. 1, Extended Data Figs 2–4 |
| `fig4_reconstruction_AMOC45N.png` | Fig. 4 |
| `fig5_EWS_AMOC45N.png` | Fig. 5 (rolling lag-1 autocorrelation and variance, 30-yr window) |

### Main options

| Option | Default | Use |
|---|---|---|
| `USE_PAPER_FEATURES` | `False` | `True` uses the predictor sets of Extended Data Table 1 instead of the search |
| `AMOC_LAT` | `"45N"` | `"26N"` for the Supplementary analysis |
| `MEMBER` | `None` | `None` uses the ensemble mean; an integer analyses a single member. Check that member order matches between AMOC and T/S |
| `DETREND_FIELDS` | `False` | Detrends T/S at each grid point before the EOF (see notes) |
| `DETREND_SERIES` | `True` | Detrends PCs and AMOC with a training-period linear fit before the regression |
| `MIN_LAG`, `MAX_LAG` | 3, 20 | Lag range (years, PC leading AMOC) |
| `N_CANDIDATES`, `K_MAX` | 10, 3 | Search size |
| `FIG_MODEL` | `"CESM2"` | Model shown in Fig. 1 (`for m in MODELS: plot_eof_pc(m)` for all) |

The lagged correlations of Extended Data Fig. 5 are available in `RES[model]["ranking"]`.

## Notes and differences from the repository code

- **Field detrending.** The paper states that linear trends are removed before the EOF analysis. `Paper_notebook.ipynb`, however, only removes the time mean. `DETREND_FIELDS = False` follows the code. Switching it on can change the EOF mode numbering, so the predictor sets in `PAPER_SETS` may then no longer match.
- **Exact match with Table 1.** None of the saved outputs in the repository reproduces Table 1 exactly; for example, CESM2 uses lags 3/4/7 with a minimum lag of 3. The defaults (`MIN_LAG = 3`, detrended series, top-10 candidates) are the closest configuration. `USE_PAPER_FEATURES = True` reproduces the published predictor sets directly.
- **Candidate filtering.** The lag filter is applied *before* taking the top-N candidates. The repository did it after, which could leave fewer than N candidates.
- **EOF centring.** EOFs are centred on the training mean, both for fitting and for projection. This only shifts PCs by a constant, so it has no effect on the regression or the correlations.
- **Not included.** The surface-only EOF analysis (Supplementary Section 5) and the Fig. 2 map are not covered.
