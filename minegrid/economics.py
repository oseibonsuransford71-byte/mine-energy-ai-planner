"""Annualised cost model. All prices are explicit assumptions, easy to change and stress-test."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CostParams:
    pv_capex_per_kwp: float = 800.0  # USD/kWp installed, utility scale
    pv_om_per_kwp_yr: float = 12.0
    pv_life_yr: int = 25
    bess_capex_per_kwh: float = 280.0  # USD/kWh installed, turnkey
    bess_om_frac: float = 0.02  # of capex per year
    bess_life_yr: int = 12
    discount_rate: float = 0.10
    diesel_price_per_l: float = 1.00  # delivered to site
    diesel_om_per_mwh: float = 20.0  # variable genset O&M
    co2_kg_per_l: float = 2.68

    def with_(self, **kw) -> "CostParams":
        return replace(self, **kw)


def crf(rate: float, years: int) -> float:
    """Capital recovery factor."""
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def capex_usd(pv_mwp: float, bess_mwh: float, c: CostParams) -> float:
    return pv_mwp * 1000.0 * c.pv_capex_per_kwp + bess_mwh * 1000.0 * c.bess_capex_per_kwh


def annual_costs(summary: dict, pv_mwp: float, bess_mwh: float, c: CostParams) -> dict:
    """Annualised cost components (USD/yr) for one simulated design."""
    pv_fixed = pv_mwp * 1000.0 * (c.pv_capex_per_kwp * crf(c.discount_rate, c.pv_life_yr) + c.pv_om_per_kwp_yr)
    bess_fixed = bess_mwh * 1000.0 * c.bess_capex_per_kwh * (crf(c.discount_rate, c.bess_life_yr) + c.bess_om_frac)
    fuel = summary["fuel_l"] * c.diesel_price_per_l
    diesel_om = summary["diesel_mwh"] * c.diesel_om_per_mwh
    total = pv_fixed + bess_fixed + fuel + diesel_om
    return {
        "pv_fixed": pv_fixed,
        "bess_fixed": bess_fixed,
        "fuel_cost": fuel,
        "diesel_om": diesel_om,
        "total": total,
        "lcoe_usd_mwh": total / summary["load_mwh"],
        "co2_t": summary["fuel_l"] * c.co2_kg_per_l / 1000.0,
        "capex": capex_usd(pv_mwp, bess_mwh, c),
    }
