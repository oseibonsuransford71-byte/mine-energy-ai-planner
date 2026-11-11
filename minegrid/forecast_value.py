"""Value of forecasting: what do forecast errors cost in day-ahead genset commitment?

Diesel gensets burn fuel for every MW of *online* capacity even when part-loaded (the no-load
term of the fuel curve). The operator decides the evening before how many units to keep
online each hour. Commit too many and fuel is wasted; commit too few and an emergency start
is needed. Better PV and load forecasts should therefore save fuel.

Scenarios (all evaluated on the 2023 test year with the same real-time dispatch):
- Oracle: commitment planned from perfect knowledge of PV and load (the floor).
- ML forecast: plan from the gradient-boosting day-ahead PV and load forecasts.
- Persistence: plan assuming tomorrow repeats today (same hour, 24 h earlier).
- No forecast: plan assuming no solar contribution at all (commit for the whole load).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import soiling_factor
from .dispatch import DispatchParams, diesel_online_capacity, simulate
from .economics import CostParams
from .solar import pv_specific_power

COMMIT_MARGIN = 0.05  # safety margin on planned diesel output when committing units


def _commit(diesel_plan_mw: np.ndarray, p: DispatchParams) -> np.ndarray:
    return diesel_online_capacity(diesel_plan_mw * (1.0 + COMMIT_MARGIN), p.unit_mw, p.max_loading)


def commitment_value(df: pd.DataFrame, preds_solar: dict, preds_load: dict, params: DispatchParams, costs: CostParams) -> pd.DataFrame:
    idx = preds_load[("load", "Gradient boosting", 24)].index
    d = df.loc[idx]
    pv_cf, load = d["pv_cf"].to_numpy(), d["load_mw"].to_numpy()

    temp_hat = df["temp"].shift(24).reindex(idx).bfill().to_numpy()
    pv_hat_persist = df["pv_cf"].shift(24).reindex(idx).bfill().to_numpy()
    load_hat_persist = df["load_mw"].shift(24).reindex(idx).bfill().to_numpy()

    _, act = simulate(pv_cf, load, params, idx, detail=True)
    diesel_actual = act["diesel"].to_numpy()
    required = diesel_online_capacity(diesel_actual, params.unit_mw, params.max_loading)

    plans = {"Oracle (perfect foresight)": diesel_actual}
    for model in ("Gradient boosting", "Ridge"):
        # day-ahead GHI forecast (daylight only) -> PV power with lagged temperature
        ghi_hat = preds_solar[("solar", model, 24)].reindex(idx).fillna(0.0).to_numpy()
        pv_hat = pv_specific_power(ghi_hat, temp_hat, soiling_factor(idx))
        load_hat = preds_load[("load", model, 24)].reindex(idx).to_numpy()
        plans[f"ML forecast ({model.lower()})"] = simulate(pv_hat, load_hat, params, idx, detail=True)[1]["diesel"].to_numpy()
    plans["Persistence (same hour yesterday)"] = simulate(pv_hat_persist, load_hat_persist, params, idx, detail=True)[1]["diesel"].to_numpy()
    plans["No forecast (assume zero solar)"] = preds_load[("load", "Gradient boosting", 24)].reindex(idx).to_numpy()

    rows = []
    for name, plan in plans.items():
        committed = _commit(plan, params)
        online = np.maximum(committed, required)  # emergency starts cover any shortfall
        fuel = float(np.sum(params.f0_l_per_mw_h * online + params.f1_l_per_mwh * diesel_actual))
        rows.append(
            {
                "scenario": name,
                "fuel_ml_per_yr": fuel / 1e6,
                "under_commit_hours": int(np.sum(committed < required)),
                "over_commit_mwh": float(np.sum(np.maximum(committed - required, 0.0))),
                "plan_rmse_mw": float(np.sqrt(np.mean((plan - diesel_actual) ** 2))),
            }
        )
    out = pd.DataFrame(rows)
    oracle = out.loc[0, "fuel_ml_per_yr"]
    years = len(idx) / 8760.0
    out["fuel_ml_per_yr"] = out["fuel_ml_per_yr"] / years
    oracle = oracle / years
    out["excess_vs_oracle_ml"] = out["fuel_ml_per_yr"] - oracle
    out["excess_vs_oracle_kusd"] = out["excess_vs_oracle_ml"] * 1e3 * costs.diesel_price_per_l
    return out
