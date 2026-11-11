#!/usr/bin/env python3
"""Run the whole project end to end: data -> forecasts -> sizing -> value of forecasting -> figures.

Usage
-----
    python scripts/run_pipeline.py                       # simulated weather (default)
    python scripts/run_pipeline.py --weather data/raw/nasa_power_weather.csv   # real irradiance

Outputs go to results/ (CSV/JSON tables) and results/figures/ (PNG).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from minegrid import plots  # noqa: E402
from minegrid.data import build_dataset, read_weather_file  # noqa: E402
from minegrid.dispatch import DispatchParams, simulate  # noqa: E402
from minegrid.economics import CostParams  # noqa: E402
from minegrid.forecast import evaluate_load, evaluate_solar  # noqa: E402
from minegrid.forecast_value import commitment_value  # noqa: E402
from minegrid.optimize import best_design, cost_vs_renewables, price_grid, sensitivity, simulate_grid  # noqa: E402

PHASE1_RE_CAP = 0.40  # "phase 1" design: cheapest system with at most 40% renewable share


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weather", help="CSV (time, ghi, temp) or NASA POWER JSON with hourly irradiance; default: simulated")
    ap.add_argument("--out", default=str(ROOT / "results"), help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # 1. data -----------------------------------------------------------------------------
    weather = read_weather_file(args.weather) if args.weather else None
    df = build_dataset(weather=weather)
    df.to_csv(ROOT / "data" / "processed" / "dataset.csv", index_label="time")
    daily = df["ghi"].resample("D").sum() / 1000.0
    summary["data"] = {
        "weather_source": args.weather or "simulated (physics-based Markov cloud model)",
        "hours": int(len(df)),
        "period": [str(df.index[0]), str(df.index[-1])],
        "ghi_kwh_m2_day": round(float(daily.mean()), 2),
        "pv_specific_yield_kwh_kwp_yr": round(float(df["pv_cf"].sum() / (len(df) / 8760)), 0),
        "load_mean_mw": round(float(df["load_mw"].mean()), 2),
        "load_peak_mw": round(float(df["load_mw"].max()), 2),
        "load_gwh_yr": round(float(df["load_mw"].sum() / (len(df) / 8760) / 1000), 1),
    }
    log(f"dataset: {summary['data']}")

    # 2. forecasting ---------------------------------------------------------------------------
    log("training solar forecasters")
    solar_metrics, solar_preds, coverage = evaluate_solar(df)
    log("training load forecasters")
    load_metrics, load_preds = evaluate_load(df)
    solar_metrics.round(4).to_csv(out / "forecast_solar_metrics.csv", index=False)
    load_metrics.round(4).to_csv(out / "forecast_load_metrics.csv", index=False)
    coverage.round(2).to_csv(out / "forecast_interval_coverage.csv", index=False)
    summary["forecast_solar"] = solar_metrics.round(4).to_dict("records")
    summary["forecast_load"] = load_metrics.round(4).to_dict("records")
    summary["interval_coverage"] = coverage.round(2).to_dict("records")

    # 3. sizing ------------------------------------------------------------------------------------
    log("simulating PV x battery design grid")
    base = DispatchParams()
    costs = CostParams()
    grid = simulate_grid(df, base)
    priced = price_grid(grid, costs)
    priced.round(4).to_csv(out / "design_grid.csv", index=False)
    best = best_design(priced)
    diesel_only = priced[(priced.pv_mwp == 0) & (priced.bess_mwh == 0)].iloc[0]
    frontier = cost_vs_renewables(priced)
    frontier.round(4).to_csv(out / "staged_frontier.csv", index=False)
    summary["diesel_only"] = {
        "fuel_ml_yr": round(diesel_only["fuel_l"] / 1e6, 2),
        "annual_cost_musd": round(diesel_only["total"] / 1e6, 2),
        "cost_usd_mwh": round(diesel_only["lcoe_usd_mwh"], 1),
        "co2_kt_yr": round(diesel_only["co2_t"] / 1e3, 1),
    }
    keys = ["pv_mwp", "bess_mwh", "ren_fraction", "curtail_fraction", "batt_cycles", "payback_yr", "capex", "total", "saving_usd_yr", "lcoe_usd_mwh", "co2_avoided_t"]
    summary["cost_optimal"] = {k: round(float(best[k]), 4) for k in keys}
    p1_row = frontier.loc[np.isclose(frontier["cap"], PHASE1_RE_CAP)].iloc[0]
    phase1 = priced[(priced.pv_mwp == p1_row.pv_mwp) & (priced.bess_mwh == p1_row.bess_mwh)].iloc[0]
    summary["phase1"] = {k: round(float(phase1[k]), 4) for k in keys}
    log(f"cost-optimal: {best.pv_mwp:.0f} MWp + {best.bess_mwh:.0f} MWh, saves ${best.saving_usd_yr/1e6:.1f}M/yr")
    log(f"phase 1: {phase1.pv_mwp:.0f} MWp + {phase1.bess_mwh:.0f} MWh, saves ${phase1.saving_usd_yr/1e6:.1f}M/yr")

    log("sensitivity analysis")
    sens = sensitivity(df, base, costs)
    sens.round(4).to_csv(out / "sensitivity.csv", index=False)

    # 4. value of forecasting ---------------------------------------------------------------------
    log("value-of-forecasting experiment")
    p1_params = base.with_(pv_mwp=float(phase1.pv_mwp), bess_mwh=float(phase1.bess_mwh))
    value = commitment_value(df, solar_preds, load_preds, p1_params, costs)
    value.round(4).to_csv(out / "forecast_value.csv", index=False)
    summary["forecast_value"] = value.round(4).to_dict("records")

    # 5. figures ---------------------------------------------------------------------------------
    log("drawing figures")
    _, det = simulate(df["pv_cf"].to_numpy(), df["load_mw"].to_numpy(), p1_params, df.index, detail=True)
    plots.fig_dispatch(
        det.loc["2023-01-09":"2023-01-15 23:00"],
        det.loc["2023-07-10":"2023-07-16 23:00"],
        figs / "fig1_dispatch.png",
        f"{p1_params.pv_mwp:.0f} MWp PV + {p1_params.bess_mwh:.0f} MWh battery + existing diesel",
    )
    plots.fig_forecast_skill(solar_metrics, figs / "fig2_solar_forecast_error.png")
    plots.fig_solar_example(solar_preds, "2023-06-05", 4, figs / "fig3_solar_forecast_example.png")
    plots.fig_load_example(load_preds, "2023-03-06", figs / "fig4_load_forecast_example.png")
    plots.fig_design_space(priced, best, figs / "fig5_design_space.png")
    plots.fig_frontier(frontier, figs / "fig6_staged_frontier.png")
    plots.fig_sensitivity(sens, figs / "fig7_sensitivity.png")
    plots.fig_forecast_value(value, figs / "fig8_forecast_value.png")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done. results in {out}")


if __name__ == "__main__":
    main()
