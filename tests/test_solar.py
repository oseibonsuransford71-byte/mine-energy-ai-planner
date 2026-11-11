import numpy as np
import pandas as pd

from minegrid.data import build_dataset
from minegrid.solar import clear_sky_ghi, pv_specific_power, solar_geometry


def test_sun_is_overhead_at_equinox_noon_on_equator():
    idx = pd.date_range("2023-03-21", periods=24, freq="h")
    cosz, _ = solar_geometry(idx, 0.0, 0.0, 0.0)
    assert cosz.max() > 0.995  # zenith angle under ~6 degrees at solar noon


def test_clear_sky_is_zero_at_night_and_bounded_by_solar_constant():
    idx = pd.date_range("2023-06-21", periods=24, freq="h")
    cosz, _ = solar_geometry(idx, 6.2, -1.67, 0.0)
    cs = clear_sky_ghi(cosz)
    assert cs[0] == 0.0 and cs[23] == 0.0
    assert 0.0 < cs.max() < 1100.0


def test_day_length_follows_latitude_seasonality():
    # at 50N, summer days are much longer than winter days; near the equator they are similar
    summer = pd.date_range("2023-06-21", periods=24, freq="h")
    winter = pd.date_range("2023-12-21", periods=24, freq="h")
    hrs = lambda idx, lat: int((solar_geometry(idx, lat, 0.0, 0.0)[0] > 0).sum())
    assert hrs(summer, 50.0) - hrs(winter, 50.0) >= 6
    assert abs(hrs(summer, 6.2) - hrs(winter, 6.2)) <= 1


def test_pv_power_is_bounded_monotonic_and_temperature_sensitive():
    ghi = np.linspace(0, 1100, 12)
    p = pv_specific_power(ghi, 25.0)
    assert p.min() >= 0.0 and p.max() <= 1.0 / 1.2 + 1e-12  # inverter clipping
    assert np.all(np.diff(p) >= -1e-12)
    assert pv_specific_power(700.0, 45.0) < pv_specific_power(700.0, 15.0)


def test_simulated_climate_is_plausible_for_southern_ghana():
    df = build_dataset("2021-01-01 00:00", "2021-12-31 23:00")
    daily_kwh = df["ghi"].resample("D").sum().mean() / 1000.0
    assert 4.0 < daily_kwh < 5.6
    yield_kwh_kwp = df["pv_cf"].sum()
    assert 1200 < yield_kwh_kwp < 1700
    assert df["load_mw"].max() < 24.0  # must stay within installed diesel capacity
    assert not df.isna().any().any()
