import numpy as np
import pytest

from minegrid.data import build_dataset
from minegrid.dispatch import DispatchParams, diesel_fuel_l, simulate
from minegrid.economics import CostParams, annual_costs, crf


@pytest.fixture(scope="module")
def month():
    return build_dataset("2022-03-01 00:00", "2022-03-31 23:00")


def run(month, **kw):
    p = DispatchParams(**kw)
    return p, *simulate(month["pv_cf"].to_numpy(), month["load_mw"].to_numpy(), p, month.index, detail=True)


def test_diesel_only_serves_the_whole_load(month):
    _, summary, d = run(month)
    assert np.allclose(d["diesel"], d["load"])
    assert summary["ren_fraction"] == 0.0


def test_load_balance_holds_every_hour(month):
    _, _, d = run(month, pv_mwp=40, bess_mwh=60)
    assert np.allclose(d["load"], d["pv_used"] + d["batt_discharge"] + d["diesel"], atol=1e-9)


def test_pv_energy_is_conserved(month):
    p, _, d = run(month, pv_mwp=40, bess_mwh=60)
    pv = month["pv_cf"].to_numpy() * p.pv_mwp
    assert np.allclose(pv, d["pv_used"] + d["batt_charge"] + d["curtailed"], atol=1e-9)


def test_battery_limits_are_respected(month):
    p, _, d = run(month, pv_mwp=40, bess_mwh=60)
    assert d["soc"].min() >= p.soc_min - 1e-9 and d["soc"].max() <= p.soc_max + 1e-9
    p_max = p.bess_mwh / p.bess_hours
    assert d["batt_charge"].max() <= p_max + 1e-9 and d["batt_discharge"].max() <= p_max + 1e-9
    assert not np.any((d["batt_charge"] > 0) & (d["batt_discharge"] > 0))


def test_minimum_diesel_share_is_enforced(month):
    p, _, d = run(month, pv_mwp=80, bess_mwh=120, min_diesel_share=0.3)
    assert np.all(d["diesel"] >= p.min_diesel_share * d["load"] - 1e-9)


def test_more_pv_never_increases_fuel(month):
    fuels = [run(month, pv_mwp=pv)[1]["fuel_l"] for pv in (0, 10, 20, 40, 80)]
    assert all(b <= a + 1e-6 for a, b in zip(fuels, fuels[1:]))


def test_round_trip_efficiency_loses_energy(month):
    _, s, d = run(month, pv_mwp=60, bess_mwh=100)
    assert d["batt_discharge"].sum() < d["batt_charge"].sum()


def test_fuel_curve_has_no_load_penalty():
    p = DispatchParams()
    # one 2 MW unit online either way: part-loading it burns more fuel per MWh
    per_mwh_part_load = diesel_fuel_l(0.5, p) / 0.5
    per_mwh_full_load = diesel_fuel_l(1.7, p) / 1.7
    assert per_mwh_part_load > per_mwh_full_load
    assert diesel_fuel_l(0.0, p) == 0.0  # no engine online, no fuel


def test_capital_recovery_factor_matches_textbook_value():
    assert crf(0.10, 20) == pytest.approx(0.117460, abs=1e-5)


def test_annual_cost_components_add_up(month):
    p, s, _ = run(month, pv_mwp=20, bess_mwh=20)
    c = annual_costs(s, 20, 20, CostParams())
    assert c["total"] == pytest.approx(c["pv_fixed"] + c["bess_fixed"] + c["fuel_cost"] + c["diesel_om"])
