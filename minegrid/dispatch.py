"""Hourly dispatch of a PV + battery + diesel system serving an off-grid mine load.

Rule-based, load-following strategy:
1. PV serves the load first.
2. Surplus PV charges the battery; whatever the battery cannot absorb is curtailed.
3. The battery discharges to cover the remaining deficit.
4. Diesel gensets cover the rest.

A minimum diesel share (spinning-reserve proxy) caps instantaneous renewable penetration at
(1 - min_diesel_share) of the load. Diesel fuel uses an affine curve with a no-load term on the
*online* capacity, which is why running part-loaded gensets is wasteful.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DispatchParams:
    pv_mwp: float = 0.0
    bess_mwh: float = 0.0
    bess_hours: float = 2.0  # energy / power ratio of the battery
    eta_roundtrip: float = 0.90
    soc_min: float = 0.10
    soc_max: float = 0.95
    min_diesel_share: float = 0.30
    unit_mw: float = 2.0  # genset unit size
    max_loading: float = 0.85  # max loading of online gensets
    f0_l_per_mw_h: float = 84.15  # no-load fuel per MW of online capacity
    f1_l_per_mwh: float = 246.0  # marginal fuel per MWh generated

    def with_(self, **kw) -> "DispatchParams":
        return replace(self, **kw)


def diesel_online_capacity(p_mw, unit_mw: float, max_loading: float):
    """Online genset capacity (MW) needed to serve p_mw without exceeding max_loading."""
    p = np.maximum(np.asarray(p_mw, dtype=float), 0.0)
    return np.ceil(p / (max_loading * unit_mw) - 1e-9) * unit_mw


def diesel_fuel_l(p_mw, params: DispatchParams, online_mw=None):
    """Fuel burned (litres per hour-step) for generation p_mw. online_mw overrides the capacity."""
    p = np.asarray(p_mw, dtype=float)
    cap = diesel_online_capacity(p, params.unit_mw, params.max_loading) if online_mw is None else online_mw
    return params.f0_l_per_mw_h * cap + params.f1_l_per_mwh * p


def simulate(pv_cf, load_mw, params: DispatchParams, index=None, detail: bool = False):
    """Run the dispatch over the supplied series (1-hour steps).

    Returns (summary, detail_df). Summary values are annual averages (MWh/yr, L/yr).
    """
    pv_cf = np.asarray(pv_cf, dtype=float)
    load = np.asarray(load_mw, dtype=float)
    n = len(load)
    years = n / 8760.0

    pv = pv_cf * params.pv_mwp
    ren_cap = (1.0 - params.min_diesel_share) * load
    pv_direct = np.minimum(np.minimum(pv, load), ren_cap)
    surplus = pv - pv_direct
    batt_allow = np.minimum(load - pv_direct, np.maximum(ren_cap - pv_direct, 0.0))

    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc = np.zeros(n)
    e_cap = params.bess_mwh
    if e_cap > 0.0:
        eta = np.sqrt(params.eta_roundtrip)
        e_min, e_max = params.soc_min * e_cap, params.soc_max * e_cap
        p_max = e_cap / params.bess_hours
        e = 0.5 * e_cap
        ch, dis, so = charge.tolist(), discharge.tolist(), soc.tolist()
        sur, allow = surplus.tolist(), batt_allow.tolist()
        for i in range(n):
            s = sur[i]
            if s > 0.0:
                c = min(s, p_max, (e_max - e) / eta)
                if c > 0.0:
                    e += c * eta
                    ch[i] = c
            else:
                d = min(allow[i], p_max, (e - e_min) * eta)
                if d > 0.0:
                    e -= d / eta
                    dis[i] = d
            so[i] = e / e_cap
        charge, discharge, soc = np.array(ch), np.array(dis), np.array(so)

    curtail = surplus - charge
    diesel = load - pv_direct - discharge
    fuel = diesel_fuel_l(diesel, params)
    served_by_re = pv_direct + discharge

    summary = {
        "load_mwh": load.sum() / years,
        "pv_mwh": pv.sum() / years,
        "re_to_load_mwh": served_by_re.sum() / years,
        "curtailed_mwh": curtail.sum() / years,
        "diesel_mwh": diesel.sum() / years,
        "fuel_l": fuel.sum() / years,
        "ren_fraction": served_by_re.sum() / load.sum(),
        "curtail_fraction": curtail.sum() / pv.sum() if pv.sum() > 0 else 0.0,
        "batt_cycles": discharge.sum() / years / (e_cap * (params.soc_max - params.soc_min)) if e_cap > 0 else 0.0,
    }
    df = None
    if detail:
        df = pd.DataFrame(
            {
                "load": load,
                "pv_used": pv_direct,
                "batt_discharge": discharge,
                "batt_charge": charge,
                "diesel": diesel,
                "curtailed": curtail,
                "soc": soc,
            },
            index=index,
        )
    return summary, df
