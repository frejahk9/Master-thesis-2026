#!/usr/bin/env python3
"""
AMOC RECONSTRUCTION + ROBUST FEATURE DISCOVERY PIPELINE (PRODUCTION)
===================================================================

Goal:
  - Reconstruct AMOC with as few interpretable features as possible
  - Choose hyperparameters using time-series CV inside train (no test leakage)
  - Identify "robust" predictors via stability selection across members
  - Optionally compute early-warning metrics (lag-1 AC, variance) on selected PCs

Supports:
  ✅ Any model with:
      - EOF_latdepth_<var>_<lon>.nc containing PC(member?, time, mode)
      - AMOC_<MODEL>.nc containing AMOC_26N_* and/or AMOC_45N_* (ensmean and member)
  ✅ Run on ensemble-mean OR member-by-member (choose)
  ✅ Final test block is last 20% of years (blocked, never used in CV/ranking)

Key design choices (thesis-defensible):
  - Feature ranking (corr across lags) is done ONLY on fold-train (inside CV)
  - Hyperparameters (N_seeds, W, alpha) chosen by mean CV validation R²
  - "Parsimonious model": choose smallest within 1σ of best CV score (optional)
  - Stability selection: keep only features appearing in ≥freq threshold across members

Outputs (in OUTDIR/<MODEL>/<TARGET>/):
  - cv_grid_results_*.csv
  - best_params_*.txt
  - best_features_*.csv
  - (member mode) member_best_features_*.csv, stability_features_*.csv
  - final_test_metrics_*.csv, final_test_predictions_*.csv
  - (optional) ews_metrics_*.csv

-------------------------------------------------------------------
USAGE EXAMPLES
-------------------------------------------------------------------

1) Ensemble mean, single model:
  python amoc_pipeline.py --model IPSL-CM6A-LR --target AMOC_45N_ensmean --mode ensmean

2) Member-by-member stability selection:
  python amoc_pipeline.py --model IPSL-CM6A-LR --target AMOC_45N_member --mode member \
      --members 0 1 2 3 4 --freq_thresh 0.6

3) All models in a list:
  python amoc_pipeline.py --models IPSL-CM6A-LR CESM2 MPI-ESM1-2-LR --target AMOC_45N_ensmean --mode ensmean

4) Turn on EWS computation for selected predictors:
  python amoc_pipeline.py --model IPSL-CM6A-LR --target AMOC_45N_member --mode member \
      --members 0 1 2 3 4 --compute_ews 1 --ews_window 31 --ews_detrend 1

-------------------------------------------------------------------
NOTES
-------------------------------------------------------------------
- If your AMOC file uses different variable names, adjust load_amoc().
- If your EOF files store time as 'year' already, this still works.
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score


# ============================================================
# I/O HELPERS
# ============================================================

def extract_years(time_coord):
    """Robust year extraction for cftime/numpy datetime or string-like."""
    try:
        return xr.DataArray(time_coord).dt.year.values.astype(int)
    except Exception:
        t = np.asarray(time_coord)
        return np.array([int(str(x)[:4]) for x in t], dtype=int)


def load_pc(EOF_DIR, var, lon_tag, n_modes, member_mode=False, member_id=None):
    """
    Loads PC from EOF_latdepth_<var>_<lon>.nc
      PC dims: (member?, time, mode) or (time, mode)
    Returns DataArray (year, mode) for ensmean OR (year, mode) for one member.
    """
    f = os.path.join(EOF_DIR, f"EOF_latdepth_{var}_{lon_tag}.nc")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing EOF file: {f}")

    ds = xr.open_dataset(f)
    PC = ds["PC"].isel(mode=slice(0, n_modes))

    if "member" in PC.dims:
        if member_mode:
            if member_id is None:
                raise ValueError("member_id must be provided for member_mode=True")
            PC = PC.sel(member=member_id)
        else:
            PC = PC.mean("member")

    PC = PC.transpose("time", "mode")
    years = extract_years(PC["time"])
    PC = PC.assign_coords(year=("time", years)).swap_dims({"time": "year"}).drop_vars("time")
    return PC.astype(float)  # (year, mode)


def load_amoc(AMOC_FILE, target, member_mode=False, member_id=None):
    """
    Loads AMOC target from AMOC_<MODEL>.nc.

    Expected possibilities (you can adapt to your own file):
      - AMOC_45N_ensmean(year)
      - AMOC_45N_member(member, year/time)

    This function returns 1D DataArray (year,) for a chosen member or ensmean.
    """
    dsA = xr.open_dataset(AMOC_FILE)
    if target not in dsA.variables:
        raise KeyError(f"Target '{target}' not found in {AMOC_FILE}. Available: {list(dsA.variables)}")

    am = dsA[target].squeeze()

    if "member" in am.dims:
        if not member_mode:
            # if user asked ensmean but file is member-only, take mean
            am = am.mean("member")
        else:
            if member_id is None:
                raise ValueError("member_id must be provided for member_mode=True")
            am = am.sel(member=member_id)

    # to year axis
    if "year" in am.dims:
        am_year = am
        if "time" in am_year.coords:
            am_year = am_year.drop_vars("time")
    else:
        years = extract_years(am["time"])
        am_year = am.assign_coords(year=("time", years)).swap_dims({"time": "year"}).drop_vars("time")

    am_year = am_year.astype(float).squeeze()
    if am_year.ndim != 1:
        raise ValueError(f"AMOC target not 1D after selection: shape={am_year.shape}")

    return am_year  # (year,)


# ============================================================
# MATH HELPERS
# ============================================================

def corr_1d(a, b):
    """Pearson correlation with NaN handling."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom == 0:
        return np.nan
    return float(np.sum(aa * bb) / denom)


def rank_features_train_only(pc_np_sub, y_sub, vars_, lons_, n_modes, lags_rank):
    """
    Rank features by |corr| using ONLY the provided subset arrays.

    pc_np_sub: dict[(var,lon)] -> np.ndarray (Tsub, n_modes)
    y_sub: np.ndarray (Tsub,)
    returns: DataFrame sorted by abs_corr desc with columns: var, lon, mode(1-based), lag, corr, abs_corr
    """
    rows = []
    Tsub = len(y_sub)
    for var in vars_:
        for lon in lons_:
            pc = pc_np_sub[(var, lon)]  # (Tsub, modes)
            for lag in lags_rank:
                lag = int(lag)
                if lag >= Tsub:
                    continue
                t_idx = np.arange(lag, Tsub)
                p_idx = t_idx - lag
                yy = y_sub[t_idx]
                for m in range(n_modes):
                    pp = pc[p_idx, m]
                    c = corr_1d(pp, yy)
                    if np.isfinite(c):
                        rows.append({
                            "var": var,
                            "lon": lon,
                            "mode": int(m + 1),
                            "lag": int(lag),
                            "corr": float(c),
                            "abs_corr": float(abs(c)),
                        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No finite correlations computed in ranking step.")
    return df.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def build_feature_set(df_ranked, n_seeds, W, max_lag_allowed, cap_lags_per_group=None):
    """
    Take top n_seeds from df_ranked and expand lag±W.
    Returns sorted unique list of (var, lon, mode0, lag).
    """
    df_top = df_ranked.head(int(n_seeds))

    seeds = []
    for _, row in df_top.iterrows():
        seeds.append((str(row["var"]), str(row["lon"]), int(row["mode"]) - 1, int(row["lag"])))

    expanded = []
    for (var, lon, m0, lag0) in seeds:
        for L in range(lag0 - int(W), lag0 + int(W) + 1):
            if L < 0:
                continue
            if max_lag_allowed is not None and L > int(max_lag_allowed):
                continue
            expanded.append((var, lon, m0, int(L)))

    if cap_lags_per_group is not None:
        keep = []
        seen = {}
        for feat in expanded:
            key = (feat[0], feat[1], feat[2])  # (var,lon,mode0)
            seen.setdefault(key, set())
            if feat[3] not in seen[key]:
                if len(seen[key]) < int(cap_lags_per_group):
                    seen[key].add(feat[3])
                    keep.append(feat)
        expanded = keep

    return sorted(set(expanded), key=lambda x: (x[0], x[1], x[2], x[3]))


def build_XY(pc_np_full, y_full, feats, t_positions):
    """
    Build X,Y using features with lags.
    pc_np_full: dict[(var,lon)] -> np.ndarray (T, n_modes)
    y_full: np.ndarray (T,)
    feats: list (var, lon, mode0, lag)
    t_positions: indices into full arrays
    Returns (X, Y, t_used, max_lag)
    """
    t_positions = np.asarray(t_positions, dtype=int)
    if len(feats) == 0:
        raise ValueError("Empty feature set.")

    max_lag = max(lag for (_, _, _, lag) in feats)
    t_used = t_positions[t_positions >= max_lag]
    if len(t_used) == 0:
        return np.empty((0, len(feats))), np.empty((0,)), t_used, int(max_lag)

    X = np.empty((len(t_used), len(feats)), dtype=float)
    for i, t in enumerate(t_used):
        for j, (var, lon, m0, lag) in enumerate(feats):
            X[i, j] = pc_np_full[(var, lon)][t - lag, m0]
    Y = y_full[t_used]
    return X, Y, t_used, int(max_lag)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if (np.std(y_true) > 0 and np.std(y_pred) > 0) else np.nan
    var_ratio = float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else np.nan
    return r2, r, var_ratio


def make_gap_folds(T_trainval, n_splits, gap):
    """
    Create expanding-window folds with a "gap/embargo" between train and val.
    This helps when using lagged predictors.

    Returns list of (train_idx, val_idx) indices relative to trainval (0..T_trainval-1).
    """
    tscv = TimeSeriesSplit(n_splits=int(n_splits))
    folds = []
    base = np.arange(T_trainval, dtype=int)
    for tr, va in tscv.split(base):
        # embargo: remove last 'gap' points from train to avoid dependence near boundary
        if gap is not None and gap > 0:
            if len(tr) <= gap:
                continue
            tr2 = tr[:-int(gap)]
        else:
            tr2 = tr
        if len(tr2) < 20 or len(va) < 10:
            continue
        folds.append((tr2, va))
    return folds


def one_se_rule_choose(df_grid, score_col="mean_val_r2", std_col="std_val_r2"):
    """
    Choose the simplest model within 1 standard deviation of the best mean score.
    "Simplest" = smaller N_seeds, smaller W, larger alpha (more regularization).
    """
    best = df_grid.iloc[0]
    threshold = best[score_col] - best[std_col]

    df_ok = df_grid[df_grid[score_col] >= threshold].copy()
    if len(df_ok) == 0:
        return best.to_dict()

    # "simplicity" ordering
    df_ok = df_ok.sort_values(
        by=["N_seeds", "W", "alpha"],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    return df_ok.iloc[0].to_dict()


# ============================================================
# EWS (OPTIONAL)
# ============================================================

def rolling_detrend(y):
    """
    Remove linear trend from a 1D array y using least squares.
    """
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    m = np.isfinite(y)
    if m.sum() < 5:
        return y * np.nan
    X = np.vstack([x[m], np.ones(m.sum())]).T
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    trend = beta[0] * x + beta[1]
    out = y.copy()
    out[m] = y[m] - trend[m]
    return out


def compute_ews_series(x, years, window=31, detrend=True):
    """
    Compute rolling lag-1 autocorrelation and variance.
    Returns DataFrame with center_year, ac1, var.

    Important: EWS on raw trending series can be biased -> detrend=True recommended.
    """
    x = np.asarray(x, float)
    years = np.asarray(years, int)
    half = window // 2
    rows = []
    for i in range(half, len(x) - half):
        seg = x[i - half:i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < max(10, window // 2):
            continue
        if detrend:
            seg = rolling_detrend(seg)
            seg = seg[np.isfinite(seg)]
            if len(seg) < max(10, window // 2):
                continue

        # lag-1 AC
        a = seg[:-1]
        b = seg[1:]
        ac1 = corr_1d(a, b)

        # variance
        vv = float(np.var(seg, ddof=1)) if len(seg) > 2 else np.nan

        rows.append({"center_year": int(years[i]), "ac1": float(ac1), "var": vv})
    return pd.DataFrame(rows)


# ============================================================
# CORE PIPELINE (ONE RUN: ONE MODEL + ONE TARGET + ONE DATASET)
# ============================================================

def run_one(model, target, mode, cfg):
    """
    mode: 'ensmean' or 'member'
    For member mode, cfg['member_id'] must be set.
    """
    EOF_DIR = cfg["EOF_DIR_TEMPLATE"].format(MODEL=model)
    AMOC_FILE = cfg["AMOC_FILE_TEMPLATE"].format(MODEL=model)

    # --- load PCs ---
    pc_dict = {}
    for var in cfg["VARS"]:
        for lon in cfg["LON_TAGS"]:
            pc_dict[(var, lon)] = load_pc(
                EOF_DIR, var, lon, cfg["N_MODES"],
                member_mode=(mode == "member"),
                member_id=cfg.get("member_id")
            )

    # --- load AMOC ---
    am = load_amoc(
        AMOC_FILE, target,
        member_mode=(mode == "member"),
        member_id=cfg.get("member_id")
    )

    # --- align years ---
    common_years = am["year"].values.astype(int)
    for da in pc_dict.values():
        common_years = np.intersect1d(common_years, da["year"].values.astype(int))
    common_years = np.asarray(common_years, int)
    common_years.sort()

    # enforce range
    y0, y1 = cfg["YEAR_START"], cfg["YEAR_END"]
    common_years = common_years[(common_years >= y0) & (common_years <= y1)]
    if len(common_years) < 50:
        raise RuntimeError(f"Too few years after applying {y0}–{y1}: T={len(common_years)}")

    y_full = am.sel(year=common_years).values.astype(float).squeeze()
    pc_np_full = {k: v.sel(year=common_years).values.astype(float) for k, v in pc_dict.items()}  # (T, mode)

    # split trainval/test (blocked last 20%)
    T = len(common_years)
    n_test = int(np.ceil(cfg["TEST_FRAC"] * T))
    n_trainval = T - n_test

    idx_trainval = np.arange(0, n_trainval, dtype=int)
    idx_test = np.arange(n_trainval, T, dtype=int)

    trainval_years = common_years[:n_trainval]
    test_years = common_years[n_trainval:]

    # lags for ranking
    max_lag_total = cfg["MAX_LAG_TOTAL"]
    max_lag_allowed = cfg["MAX_LAG_ALLOWED"]
    lags = np.arange(0, max_lag_total, dtype=int)
    if max_lag_allowed is not None:
        lags = lags[lags <= int(max_lag_allowed)]

    # folds with gap
    gap = cfg["GAP_EMBARGO"]
    folds = make_gap_folds(len(idx_trainval), cfg["N_SPLITS"], gap)
    if len(folds) < 2:
        raise RuntimeError("Not enough folds after applying gap/size constraints. Reduce GAP or N_SPLITS.")

    # grid search with fold-train ranking (no leakage)
    grid_rows = []
    for N_seeds in cfg["N_LIST"]:
        for W in cfg["W_LIST"]:
            for alpha in cfg["RIDGE_ALPHAS"]:
                fold_r2, fold_corr, fold_var = [], [], []
                fold_nfeat, fold_maxlag = [], []

                for (tr_rel, va_rel) in folds:
                    tr_idx = idx_trainval[tr_rel]
                    va_idx = idx_trainval[va_rel]

                    # ranking uses only fold-train
                    y_tr = y_full[tr_idx]
                    pc_tr = {(var, lon): pc_np_full[(var, lon)][tr_idx, :] for var in cfg["VARS"] for lon in cfg["LON_TAGS"]}
                    df_rank = rank_features_train_only(pc_tr, y_tr, cfg["VARS"], cfg["LON_TAGS"], cfg["N_MODES"], lags)

                    feats = build_feature_set(
                        df_ranked=df_rank,
                        n_seeds=N_seeds,
                        W=W,
                        max_lag_allowed=max_lag_allowed,
                        cap_lags_per_group=cfg["CAP_LAGS_PER_GROUP"]
                    )

                    Xtr, Ytr, _, maxlag = build_XY(pc_np_full, y_full, feats, tr_idx)
                    Xva, Yva, _, _      = build_XY(pc_np_full, y_full, feats, va_idx)

                    if len(Ytr) < 20 or len(Yva) < 10:
                        continue

                    mdl = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
                    mdl.fit(Xtr, Ytr)
                    pred = mdl.predict(Xva)

                    r2, r, vr = metrics(Yva, pred)
                    fold_r2.append(r2)
                    fold_corr.append(r)
                    fold_var.append(vr)
                    fold_nfeat.append(len(feats))
                    fold_maxlag.append(maxlag)

                if len(fold_r2) < max(2, len(folds) - 1):
                    continue

                grid_rows.append({
                    "model": model,
                    "target": target,
                    "mode": mode,
                    "member_id": cfg.get("member_id", np.nan),
                    "N_seeds": int(N_seeds),
                    "W": int(W),
                    "alpha": float(alpha),
                    "mean_val_r2": float(np.nanmean(fold_r2)),
                    "std_val_r2": float(np.nanstd(fold_r2)),
                    "mean_val_corr": float(np.nanmean(fold_corr)),
                    "mean_val_var_ratio": float(np.nanmean(fold_var)),
                    "mean_n_features": float(np.mean(fold_nfeat)),
                    "mean_max_lag": float(np.mean(fold_maxlag)),
                    "n_folds_used": int(len(fold_r2)),
                })

    df_grid = pd.DataFrame(grid_rows)
    if len(df_grid) == 0:
        raise RuntimeError("No CV results (grid too strict / folds too small).")

    # sort by best validation R²
    df_grid = df_grid.sort_values(["mean_val_r2", "std_val_r2"], ascending=[False, True]).reset_index(drop=True)

    # choose best params (optionally 1-SE rule for parsimony)
    if cfg["USE_ONE_SE_RULE"]:
        best = one_se_rule_choose(df_grid)
        best_tag = "best_1se"
    else:
        best = df_grid.iloc[0].to_dict()
        best_tag = "best"

    # final feature ranking on full trainval only (still no test)
    y_tv = y_full[idx_trainval]
    pc_tv = {(var, lon): pc_np_full[(var, lon)][idx_trainval, :] for var in cfg["VARS"] for lon in cfg["LON_TAGS"]}
    df_rank_tv = rank_features_train_only(pc_tv, y_tv, cfg["VARS"], cfg["LON_TAGS"], cfg["N_MODES"], lags)

    feats_final = build_feature_set(
        df_ranked=df_rank_tv,
        n_seeds=int(best["N_seeds"]),
        W=int(best["W"]),
        max_lag_allowed=max_lag_allowed,
        cap_lags_per_group=cfg["CAP_LAGS_PER_GROUP"]
    )

    # fit final model on full trainval
    X_tv, Y_tv, tv_used, maxlag_final = build_XY(pc_np_full, y_full, feats_final, idx_trainval)
    final_model = make_pipeline(StandardScaler(), Ridge(alpha=float(best["alpha"])))
    final_model.fit(X_tv, Y_tv)

    # evaluate once on test
    X_te, Y_te, te_used, _ = build_XY(pc_np_full, y_full, feats_final, idx_test)
    pred_te = final_model.predict(X_te)
    test_r2, test_corr, test_var = metrics(Y_te, pred_te)

    result = {
        "df_grid": df_grid,
        "best": best,
        "best_tag": best_tag,
        "feats_final": feats_final,
        "years": common_years,
        "trainval_years": trainval_years,
        "test_years": test_years,
        "test_metrics": {
            "test_r2": test_r2,
            "test_corr": test_corr,
            "test_var_ratio": test_var,
            "max_lag_used": maxlag_final,
            "n_features_used": len(feats_final),
        },
        "test_pred_df": pd.DataFrame({
            "year": common_years[te_used],
            "y_true": Y_te,
            "y_pred": pred_te
        })
    }

    # optional EWS on selected PC series (unlagged)
    if cfg["COMPUTE_EWS"]:
        ews_rows = []
        # build series for each unique (var,lon,mode0) in final features
        uniq = sorted(set((v, l, m0) for (v, l, m0, _) in feats_final))
        for (v, l, m0) in uniq:
            series = pc_np_full[(v, l)][:, m0]
            df_ews = compute_ews_series(series, common_years, window=cfg["EWS_WINDOW"], detrend=cfg["EWS_DETREND"])
            df_ews["model"] = model
            df_ews["target"] = target
            df_ews["mode"] = int(m0 + 1)
            df_ews["var"] = v
            df_ews["lon"] = l
            df_ews["mode_key"] = f"{v}_{l}_m{m0+1}"
            ews_rows.append(df_ews)
        result["ews_df"] = pd.concat(ews_rows, ignore_index=True) if len(ews_rows) else pd.DataFrame()

    return result


# ============================================================
# STABILITY SELECTION ACROSS MEMBERS
# ============================================================

def stability_selection(member_results, freq_thresh=0.6):
    """
    member_results: list of dict results from run_one() in member mode.
    Returns:
      - df_member_features: long table of features per member
      - df_stable: stable features with frequency >= freq_thresh
    """
    all_rows = []
    for res in member_results:
        mid = res["best"].get("member_id", np.nan)
        for (var, lon, m0, lag) in res["feats_final"]:
            all_rows.append({
                "member_id": int(mid) if np.isfinite(mid) else mid,
                "var": var,
                "lon": lon,
                "mode0": int(m0),
                "mode": int(m0 + 1),
                "lag": int(lag),
                "feature_key": f"{var}|{lon}|m{m0}|lag{lag}"
            })
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df, pd.DataFrame()

    counts = df.groupby(["var", "lon", "mode0", "lag", "feature_key"]).size().reset_index(name="count")
    n_members = df["member_id"].nunique()
    counts["freq"] = counts["count"] / float(n_members)

    df_stable = counts[counts["freq"] >= float(freq_thresh)].copy()
    df_stable = df_stable.sort_values(["freq", "count"], ascending=[False, False]).reset_index(drop=True)

    return df, df_stable


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Single model name (e.g. IPSL-CM6A-LR)")
    ap.add_argument("--models", nargs="*", default=None, help="List of models (overrides --model)")
    ap.add_argument("--target", required=True, help="AMOC variable name in AMOC_<MODEL>.nc, e.g. AMOC_45N_ensmean or AMOC_45N_member")
    ap.add_argument("--mode", choices=["ensmean", "member"], required=True, help="Run on ensmean or member-by-member")
    ap.add_argument("--members", nargs="*", type=int, default=None, help="Member IDs to run (member mode only). If omitted, tries 0..9")
    ap.add_argument("--freq_thresh", type=float, default=0.6, help="Stability selection frequency threshold across members")

    # core paths (edit if needed)
    ap.add_argument("--eof_dir_template", default="/data/projects/nckf/frekle/EOF_results/{MODEL}/latdepth_sections/")
    ap.add_argument("--amoc_file_template", default="/data/users/frekle/AMOC_analysis/AMOC_{MODEL}.nc")
    ap.add_argument("--out_root", default="/data/users/frekle/AMOC_analysis/PIPELINE_RESULTS/")

    # time + split
    ap.add_argument("--year_start", type=int, default=1850)
    ap.add_argument("--year_end", type=int, default=2014)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--gap_embargo", type=int, default=25, help="Gap between train and val inside CV (use <= max_lag_allowed).")

    # feature space
    ap.add_argument("--vars", nargs="*", default=["thetao", "so"])
    ap.add_argument("--lons", nargs="*", default=["W10p0", "W20p0", "W30p0", "W40p0", "W60p0"])
    ap.add_argument("--n_modes", type=int, default=10)
    ap.add_argument("--max_lag_total", type=int, default=50)
    ap.add_argument("--max_lag_allowed", type=int, default=25)
    ap.add_argument("--cap_lags_per_group", type=int, default=None)

    # hyperparam grids (small by default -> parsimonious)
    ap.add_argument("--N_list", nargs="*", type=int, default=[5, 10, 15])
    ap.add_argument("--W_list", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.1, 1, 10, 100, 1000])

    # parsimony rule
    ap.add_argument("--use_one_se_rule", type=int, default=1, help="1 = choose simplest within 1σ of best CV score")

    # EWS
    ap.add_argument("--compute_ews", type=int, default=0)
    ap.add_argument("--ews_window", type=int, default=31)
    ap.add_argument("--ews_detrend", type=int, default=1)

    return ap.parse_args()


def main():
    args = parse_args()

    models = args.models if args.models else ([args.model] if args.model else None)
    if not models:
        raise ValueError("Provide --model or --models")

    cfg = {
        "EOF_DIR_TEMPLATE": args.eof_dir_template,
        "AMOC_FILE_TEMPLATE": args.amoc_file_template,
        "OUT_ROOT": args.out_root,

        "YEAR_START": args.year_start,
        "YEAR_END": args.year_end,
        "TEST_FRAC": args.test_frac,
        "N_SPLITS": args.n_splits,
        "GAP_EMBARGO": args.gap_embargo,

        "VARS": args.vars,
        "LON_TAGS": args.lons,
        "N_MODES": args.n_modes,

        "MAX_LAG_TOTAL": args.max_lag_total,
        "MAX_LAG_ALLOWED": args.max_lag_allowed,
        "CAP_LAGS_PER_GROUP": args.cap_lags_per_group,

        "N_LIST": args.N_list,
        "W_LIST": args.W_list,
        "RIDGE_ALPHAS": args.alphas,

        "USE_ONE_SE_RULE": bool(args.use_one_se_rule),

        "COMPUTE_EWS": bool(args.compute_ews),
        "EWS_WINDOW": args.ews_window,
        "EWS_DETREND": bool(args.ews_detrend),
    }

    for model in models:
        outdir = os.path.join(cfg["OUT_ROOT"], model, args.target, f"mode_{args.mode}")
        os.makedirs(outdir, exist_ok=True)

        if args.mode == "ensmean":
            res = run_one(model, args.target, "ensmean", cfg)

            # save grid + best
            grid_csv = os.path.join(outdir, f"{model}_{args.target}_cv_grid_results.csv")
            res["df_grid"].to_csv(grid_csv, index=False)

            best_txt = os.path.join(outdir, f"{model}_{args.target}_{res['best_tag']}_params.txt")
            with open(best_txt, "w") as f:
                for k, v in res["best"].items():
                    f.write(f"{k}: {v}\n")

            feat_csv = os.path.join(outdir, f"{model}_{args.target}_{res['best_tag']}_features.csv")
            pd.DataFrame(res["feats_final"], columns=["var", "lon", "mode0", "lag"]).to_csv(feat_csv, index=False)

            met_csv = os.path.join(outdir, f"{model}_{args.target}_final_test_metrics.csv")
            pd.DataFrame([{
                "model": model,
                "target": args.target,
                **res["test_metrics"],
                "trainval_start": int(res["trainval_years"][0]),
                "trainval_end": int(res["trainval_years"][-1]),
                "test_start": int(res["test_years"][0]),
                "test_end": int(res["test_years"][-1]),
            }]).to_csv(met_csv, index=False)

            pred_csv = os.path.join(outdir, f"{model}_{args.target}_final_test_predictions.csv")
            res["test_pred_df"].to_csv(pred_csv, index=False)

            if cfg["COMPUTE_EWS"]:
                ews_csv = os.path.join(outdir, f"{model}_{args.target}_ews_metrics.csv")
                res["ews_df"].to_csv(ews_csv, index=False)

            print(f"\n[{model} | {args.target} | ensmean] DONE")
            print("  grid:", grid_csv)
            print("  best:", best_txt)
            print("  feats:", feat_csv)
            print("  metrics:", met_csv)
            print("  preds:", pred_csv)

        else:
            # member-by-member
            members = args.members if args.members else list(range(10))
            member_results = []

            for mid in members:
                cfg2 = dict(cfg)
                cfg2["member_id"] = mid
                print(f"\n[{model} | {args.target} | member {mid}] running...")
                try:
                    res = run_one(model, args.target, "member", cfg2)
                except Exception as e:
                    print(f"  ⚠️ member {mid} failed: {e}")
                    continue

                # save per-member artifacts
                tag = res["best_tag"]
                grid_csv = os.path.join(outdir, f"{model}_{args.target}_member{mid}_cv_grid.csv")
                res["df_grid"].to_csv(grid_csv, index=False)

                best_txt = os.path.join(outdir, f"{model}_{args.target}_member{mid}_{tag}_params.txt")
                with open(best_txt, "w") as f:
                    for k, v in res["best"].items():
                        f.write(f"{k}: {v}\n")

                feat_csv = os.path.join(outdir, f"{model}_{args.target}_member{mid}_{tag}_features.csv")
                pd.DataFrame(res["feats_final"], columns=["var", "lon", "mode0", "lag"]).to_csv(feat_csv, index=False)

                met_csv = os.path.join(outdir, f"{model}_{args.target}_member{mid}_final_test_metrics.csv")
                pd.DataFrame([{
                    "model": model,
                    "target": args.target,
                    "member_id": mid,
                    **res["test_metrics"],
                    "trainval_start": int(res["trainval_years"][0]),
                    "trainval_end": int(res["trainval_years"][-1]),
                    "test_start": int(res["test_years"][0]),
                    "test_end": int(res["test_years"][-1]),
                }]).to_csv(met_csv, index=False)

                pred_csv = os.path.join(outdir, f"{model}_{args.target}_member{mid}_final_test_predictions.csv")
                res["test_pred_df"].to_csv(pred_csv, index=False)

                if cfg["COMPUTE_EWS"]:
                    ews_csv = os.path.join(outdir, f"{model}_{args.target}_member{mid}_ews_metrics.csv")
                    res["ews_df"].to_csv(ews_csv, index=False)

                member_results.append(res)

            # stability selection across successful members
            df_member_feats, df_stable = stability_selection(member_results, freq_thresh=args.freq_thresh)

            member_feat_csv = os.path.join(outdir, f"{model}_{args.target}_member_best_features_long.csv")
            df_member_feats.to_csv(member_feat_csv, index=False)

            stable_csv = os.path.join(outdir, f"{model}_{args.target}_stability_features_freq{args.freq_thresh:.2f}.csv")
            df_stable.to_csv(stable_csv, index=False)

            print(f"\n[{model} | {args.target} | member mode] DONE")
            print("  member feats:", member_feat_csv)
            print("  stable feats:", stable_csv)
            print(f"  members_successful: {len(member_results)} / {len(members)}")


if __name__ == "__main__":
    main()
