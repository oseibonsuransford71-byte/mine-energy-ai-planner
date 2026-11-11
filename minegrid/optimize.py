"""Design-space search: size PV and battery to minimise annualised system cost."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dispatch import DispatchParams, simulate
from .economics import CostParams, annual_costs

PV_GRID = tuple(range(0, 101, 10))  # MWp
BESS_GRID = tuple(range(0, 241, 20))  # MWh


def simulate_grid(df: pd.DataFrame, base: DispatchParams, pv_grid=PV_GRID, bess_grid=BESS_GRID) -> pd.DataFrame:
    """Physical simulation of every (PV, battery) design. Independent of prices."""
    rows = []
    for pv in pv_grid:
        for bess in bess_grid:
            s, _ = simulate(df["pv_cf"].to_numpy(), df["load_mw"].to_numpy(), base.with_(pv_mwp=pv, bess_mwh=bess))
            rows.append({"pv_mwp": pv, "bess_mwh": bess, **s})
    return pd.DataFrame(rows)


def price_grid(grid: pd.DataFrame, costs: CostParams) -> pd.DataFrame:
    """Attach annualised costs, savings versus diesel-only, CO2 and payback to a simulated grid."""
    out = grid.copy()
    comp = [annual_costs(r, r["pv_mwp"], r["bess_mwh"], costs) for r in out.to_dict("records")]
    comp = pd.DataFrame(comp, index=out.index)
    out = pd.concat([out, comp], axis=1)
    base = out[(out.pv_mwp == 0) & (out.bess_mwh == 0)].iloc[0]
    out["saving_usd_yr"] = base["total"] - out["total"]
    out["co2_avoided_t"] = base["co2_t"] - out["co2_t"]
    opex_base = base["fuel_cost"] + base["diesel_om"]
    opex = out["fuel_cost"] + out["diesel_om"]
    fixed_om = out["pv_mwp"] * 1000.0 * costs.pv_om_per_kwp_yr + out["bess_mwh"] * 1000.0 * costs.bess_capex_per_kwh * costs.bess_om_frac
    net_op_saving = opex_base - opex - fixed_om
    out["payback_yr"] = np.where(net_op_saving > 0, out["capex"] / net_op_saving.where(net_op_saving > 0), np.inf)
    out.loc[(out.pv_mwp == 0) & (out.bess_mwh == 0), "payback_yr"] = np.nan
    return out


def best_design(priced: pd.DataFrame) -> pd.Series:
    return priced.loc[priced["total"].idxmin()]


def cost_vs_renewables(priced: pd.DataFrame, caps=np.arange(0.1, 0.71, 0.05)) -> pd.DataFrame:
    """Cheapest design for each cap on renewable share (a staged-investment frontier).

    Each row answers: if the mine will only commit to a renewable share of at most `cap`,
    which PV/battery size is cheapest and what does it save versus diesel-only?
    """
    rows = []
    for cap in caps:
        ok = priced[priced["ren_fraction"] <= cap + 1e-9]
        b = ok.loc[ok["total"].idxmin()]
        rows.append(
            {
                "cap": cap,
                "pv_mwp": b["pv_mwp"],
                "bess_mwh": b["bess_mwh"],
                "ren_fraction": b["ren_fraction"],
                "total_musd": b["total"] / 1e6,
                "saving_musd": b["saving_usd_yr"] / 1e6,
            }
        )
    return pd.DataFrame(rows)


def sensitivity(df: pd.DataFrame, base: DispatchParams, costs: CostParams) -> pd.DataFrame:
    """Optimal design under different diesel prices, reserve rules and PV/battery capex."""
    rows = []
    for share in (0.1, 0.3, 0.5):
        grid = simulate_grid(df, base.with_(min_diesel_share=share))
        for price in (0.6, 0.8, 1.0, 1.2, 1.4):
            for capex_mult in (0.7, 1.0, 1.3):
                c = costs.with_(
                    diesel_price_per_l=price,
                    pv_capex_per_kwp=costs.pv_capex_per_kwp * capex_mult,
                    bess_capex_per_kwh=costs.bess_capex_per_kwh * capex_mult,
                )
                p = price_grid(grid, c)
                b = best_design(p)
                rows.append(
                    {
                        "min_diesel_share": share,
                        "diesel_price": price,
                        "capex_mult": capex_mult,
                        "pv_mwp": b["pv_mwp"],
                        "bess_mwh": b["bess_mwh"],
                        "ren_fraction": b["ren_fraction"],
                        "saving_musd": b["saving_usd_yr"] / 1e6,
                        # True when the optimum sits on the edge of the search grid: the true
                        # optimum may be larger, so the saving is then a lower bound.
                        "at_grid_edge": bool(b["pv_mwp"] >= max(PV_GRID) or b["bess_mwh"] >= max(BESS_GRID)),
                    }
                )
    return pd.DataFrame(rows)
