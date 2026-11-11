"""Day-ahead and intra-day forecasting of solar irradiance and mine load.

Design rules that keep the evaluation honest
--------------------------------------------
* Every feature row is indexed by the *target* time T. A feature that uses data from the
  issue time t = T - h or earlier is built with shift(h + j), j >= 0. Nothing later is used
  (verified by tests/test_forecast_leakage.py).
* Strict chronological split: train on 2021-2022, test on 2023. No shuffling.
* Solar is forecast as the clear-sky index (kcs) and converted back to GHI with the
  deterministic clear-sky curve, so the model only has to learn *clouds*, not the sun's path.
* Baselines are the standard ones in the solar-forecasting literature: smart persistence and
  climatology (and, for load, same-hour-yesterday and same-hour-last-week).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import TRAIN_END

SOLAR_HORIZONS = (1, 6, 24)
DAYLIGHT_WM2 = 50.0  # evaluate solar forecasts only when clear-sky GHI exceeds this
QUANTILES = (0.1, 0.9)
CAL_DAYS = 120  # last days of the training period held out to calibrate prediction intervals

# columns that are deterministic functions of the target timestamp (not observations)
DETERMINISTIC_COLS = ("cosz", "cs", "sin_h", "cos_h", "sin_d", "cos_d", "sin_w", "cos_w")


def _hgb(**kw) -> HistGradientBoostingRegressor:
    params = dict(max_iter=250, learning_rate=0.06, max_depth=5, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def _calendar(idx: pd.DatetimeIndex) -> dict:
    h = np.asarray(idx.hour, dtype=float)
    d = np.asarray(idx.dayofyear, dtype=float)
    w = np.asarray(idx.dayofweek, dtype=float)
    return {
        "sin_h": np.sin(2 * np.pi * h / 24),
        "cos_h": np.cos(2 * np.pi * h / 24),
        "sin_d": np.sin(2 * np.pi * d / 365.25),
        "cos_d": np.cos(2 * np.pi * d / 365.25),
        "sin_w": np.sin(2 * np.pi * w / 7),
        "cos_w": np.cos(2 * np.pi * w / 7),
    }


def solar_features(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Features for forecasting kcs at T = t + h, using only information up to t."""
    assert 1 <= h <= 24, "horizon must be 1..24 h (yesterday's same-hour value must be observed)"
    k = df["kcs"]
    X = pd.DataFrame(index=df.index)
    X["kcs_t"] = k.shift(h)
    for j in (1, 2, 3):
        X[f"kcs_t-{j}"] = k.shift(h + j)
    X["kcs_roll6"] = k.shift(h).rolling(6).mean()
    X["kcs_roll24"] = k.shift(h).rolling(24).mean()
    X["kcs_yday_same_hour"] = k.shift(24)
    X["temp_t"] = df["temp"].shift(h)
    X["cosz"] = df["cosz"]
    X["cs"] = df["ghi_cs"]
    for name, v in _calendar(df.index).items():
        X[name] = v
    return X


def load_features(df: pd.DataFrame, h: int = 24) -> pd.DataFrame:
    """Features for forecasting load at T = t + h (default: day-ahead)."""
    assert 1 <= h <= 24
    L = df["load_mw"]
    X = pd.DataFrame(index=df.index)
    X["load_t"] = L.shift(h)
    X["load_t-1"] = L.shift(h + 1)
    X["load_yday_same_hour"] = L.shift(24)
    X["load_2d_same_hour"] = L.shift(48)
    X["load_lastweek_same_hour"] = L.shift(168)
    X["load_roll24"] = L.shift(h).rolling(24).mean()
    X["load_med168"] = L.shift(h).rolling(168).median()
    X["temp_yday"] = df["temp"].shift(24)
    for name, v in _calendar(df.index).items():
        X[name] = v
    return X


def _split(idx: pd.DatetimeIndex):
    train = idx <= pd.Timestamp(TRAIN_END)
    return train, ~train


def _metrics(y: np.ndarray, p: np.ndarray, scale: float) -> dict:
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"MAE": mae, "RMSE": rmse, "nMAE_%": 100.0 * mae / scale, "nRMSE_%": 100.0 * rmse / scale, "bias": float(err.mean())}


def evaluate_solar(df: pd.DataFrame, horizons=SOLAR_HORIZONS):
    """Train and test solar forecasters. Returns (metrics_df, predictions_dict, coverage_df).

    Metrics are in W/m2 of GHI over daylight hours of the 2023 test year. Skill is
    1 - RMSE / RMSE_smart_persistence.
    """
    rows, preds, cover = [], {}, []
    for h in horizons:
        X = solar_features(df, h)
        y = df["kcs"]
        day = (df["ghi_cs"] > DAYLIGHT_WM2) & X.notna().all(axis=1)
        train_m, test_m = _split(df.index)
        tr, te = train_m & day.to_numpy(), test_m & day.to_numpy()
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        cs_te = df["ghi_cs"][te].to_numpy()
        ghi_te = df["ghi"][te].to_numpy()
        scale = float(ghi_te.mean())

        # climatology: mean kcs by (month, hour) in training data
        clim = y[tr].groupby([df.index[tr].month, df.index[tr].hour]).mean()
        clim_te = np.array([clim.get((m, hr), y[tr].mean()) for m, hr in zip(df.index[te].month, df.index[te].hour)])

        kcs_hat = {
            "Climatology": clim_te,
            "Smart persistence": Xte["kcs_t"].to_numpy(),
            "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr).predict(Xte),
        }
        gbm = _hgb().fit(Xtr, ytr)
        kcs_hat["Gradient boosting"] = gbm.predict(Xte)

        results = {}
        for name, k_hat in kcs_hat.items():
            k_hat = np.clip(k_hat, 0.0, 1.2)
            pred_ghi = k_hat * cs_te
            results[name] = _metrics(ghi_te, pred_ghi, scale)
            preds[("solar", name, h)] = pd.Series(pred_ghi, index=df.index[te])
        base_rmse = results["Smart persistence"]["RMSE"]
        for name, m in results.items():
            rows.append({"target": "GHI (W/m2)", "horizon_h": h, "model": name, **m, "skill_vs_persistence": 1.0 - m["RMSE"] / base_rmse})

        # Prediction intervals: quantile gradient boosting, then conformalized quantile regression
        # (Romano et al., 2019). The last CAL_DAYS of the training period are held out to calibrate.
        cal_cut = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=CAL_DAYS)
        fit_m = tr & (df.index <= cal_cut)
        cal_m = tr & (df.index > cal_cut)
        cs_cal, ghi_cal = df["ghi_cs"][cal_m].to_numpy(), df["ghi"][cal_m].to_numpy()
        q_models = {q: _hgb(loss="quantile", quantile=q).fit(X[fit_m], y[fit_m]) for q in QUANTILES}

        def _band(rows_mask, cs):
            a = np.clip(q_models[QUANTILES[0]].predict(X[rows_mask]), 0, 1.2) * cs
            b = np.clip(q_models[QUANTILES[1]].predict(X[rows_mask]), 0, 1.2) * cs
            return np.minimum(a, b), np.maximum(a, b)

        lo_cal, hi_cal = _band(cal_m, cs_cal)
        scores = np.maximum(lo_cal - ghi_cal, ghi_cal - hi_cal)
        alpha = 1.0 - (QUANTILES[1] - QUANTILES[0])
        n_cal = len(scores)
        qhat = float(np.quantile(scores, min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)))
        lo_raw, hi_raw = _band(te, cs_te)
        lo, hi = np.maximum(lo_raw - qhat, 0.0), hi_raw + qhat
        preds[("solar_lo", "Gradient boosting", h)] = pd.Series(lo, index=df.index[te])
        preds[("solar_hi", "Gradient boosting", h)] = pd.Series(hi, index=df.index[te])
        preds[("solar_obs", "Observed", h)] = pd.Series(ghi_te, index=df.index[te])
        cover.append(
            {
                "horizon_h": h,
                "nominal_%": 100 * (QUANTILES[1] - QUANTILES[0]),
                "raw_quantile_%": 100.0 * float(np.mean((ghi_te >= lo_raw) & (ghi_te <= hi_raw))),
                "conformal_%": 100.0 * float(np.mean((ghi_te >= lo) & (ghi_te <= hi))),
                "conformal_width_Wm2": float(np.mean(hi - lo)),
            }
        )
    return pd.DataFrame(rows), preds, pd.DataFrame(cover)


def evaluate_load(df: pd.DataFrame, h: int = 24):
    """Day-ahead load forecasting. Returns (metrics_df, predictions_dict)."""
    X = load_features(df, h)
    y = df["load_mw"]
    ok = X.notna().all(axis=1).to_numpy()
    train_m, test_m = _split(df.index)
    tr, te = train_m & ok, test_m & ok
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    yte = y[te].to_numpy()
    scale = float(yte.mean())

    p = {
        "Same hour yesterday": Xte["load_yday_same_hour"].to_numpy(),
        "Same hour last week": Xte["load_lastweek_same_hour"].to_numpy(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr).predict(Xte),
        "Gradient boosting": _hgb().fit(Xtr, ytr).predict(Xte),
    }
    rows, preds = [], {}
    base = _metrics(yte, p["Same hour yesterday"], scale)["RMSE"]
    for name, pred in p.items():
        m = _metrics(yte, pred, scale)
        rows.append({"target": "Load (MW)", "horizon_h": h, "model": name, **m, "skill_vs_persistence": 1.0 - m["RMSE"] / base})
        preds[("load", name, h)] = pd.Series(pred, index=df.index[te])
    preds[("load_obs", "Observed", h)] = pd.Series(yte, index=df.index[te])
    return pd.DataFrame(rows), preds
