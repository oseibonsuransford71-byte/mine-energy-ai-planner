"""Synthetic weather and mine electrical load.

IMPORTANT: this is a *simulation*, not measured data. Weather follows a seasonal three-state
(clear / partly cloudy / overcast) Markov model with hourly AR(1) noise, tuned to look like the
humid-tropical wet/dry cycle of southern Ghana. Mine load is built bottom-up from the main
consumers of a gold operation (CIL processing plant, crushing, dewatering, camp). Real
irradiance can be swapped in with `minegrid.data.weather_from_nasa_power`; real mine load is
usually confidential, so a load simulator is used in any case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import DEFAULT_SITE, Site
from .solar import clear_sky_ghi, solar_geometry

# Weather states: 0 clear, 1 partly cloudy, 2 overcast
KCS_MEAN = np.array([0.90, 0.70, 0.32])
KCS_STD = np.array([0.05, 0.17, 0.08])
P_DRY = np.array([0.50, 0.40, 0.10])
P_WET = np.array([0.15, 0.45, 0.40])
PERSISTENCE = 0.55  # probability mass kept on the previous day's state


def wetness(doy: np.ndarray) -> np.ndarray:
    """0 = driest (mid January), 1 = wettest (mid July). Smooth seasonal index."""
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * (np.asarray(doy, dtype=float) - 200.0) / 365.0))


def harmattan(doy: np.ndarray) -> np.ndarray:
    """Dust-haze intensity in [0, 1], peaking around 20 January (Dec-Feb harmattan)."""
    doy = np.asarray(doy, dtype=float)
    d = np.abs(doy - 20.0)
    d = np.minimum(d, 365.0 - d)
    return np.exp(-0.5 * (d / 30.0) ** 2)


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1) noise."""
    eps = rng.standard_normal(n)
    return lfilter([np.sqrt(1.0 - phi**2)], [1.0, -phi], eps)


def synth_weather(index: pd.DatetimeIndex, site: Site = DEFAULT_SITE, seed: int = 42) -> pd.DataFrame:
    """Hourly GHI (W/m2) and ambient temperature (degC)."""
    assert index.freq is not None or len(index) > 1
    assert index[0].hour == 0, "index must start at midnight"
    rng = np.random.default_rng(seed)
    n = len(index)
    cosz, _ = solar_geometry(index, site.lat, site.lon, site.tz_hours)
    cs = clear_sky_ghi(cosz)

    n_days = int(np.ceil(n / 24))
    day_doy = np.asarray(index[::24].dayofyear)[:n_days]
    state = np.empty(n_days, dtype=int)
    state[0] = 1
    for d in range(1, n_days):
        w = wetness(day_doy[d])
        stationary = (1.0 - w) * P_DRY + w * P_WET
        trans = PERSISTENCE * np.eye(3)[state[d - 1]] + (1.0 - PERSISTENCE) * stationary
        state[d] = rng.choice(3, p=trans)
    state_h = np.repeat(state, 24)[:n]

    noise = _ar1(n, 0.85, rng)
    doy = np.asarray(index.dayofyear)
    haze = 1.0 - 0.08 * harmattan(doy)
    kcs = np.clip((KCS_MEAN[state_h] + KCS_STD[state_h] * noise) * haze, 0.02, 1.10)
    ghi = kcs * cs

    hour = np.asarray(index.hour) + 0.5
    t_mean = 26.5 + 1.5 * np.cos(2.0 * np.pi * (doy - 50.0) / 365.0)
    amp = 2.5 + 3.5 * np.clip(kcs, 0.0, 1.0)
    temp = t_mean + amp * np.cos(2.0 * np.pi * (hour - 14.5) / 24.0) + 0.6 * _ar1(n, 0.9, rng)
    return pd.DataFrame({"ghi": ghi, "temp": temp}, index=index)


def synth_load(index: pd.DatetimeIndex, temp_c: np.ndarray, seed: int = 43) -> pd.Series:
    """Hourly electrical demand (MW) of an illustrative gold mine.

    Components: CIL processing plant (~11 MW, random trips and planned maintenance), crushing and
    conveyors (shift pattern), pit dewatering (higher in the wet season), camp and ancillaries
    (evening peak, temperature-dependent cooling).
    """
    rng = np.random.default_rng(seed)
    n = len(index)
    hour = np.asarray(index.hour)
    dow = np.asarray(index.dayofweek)
    doy = np.asarray(index.dayofyear)
    wet = wetness(doy)

    # ore hardness random walk (daily) modulates mill power draw
    n_days = int(np.ceil(n / 24))
    hard_d = lfilter([0.04 * np.sqrt(1 - 0.9**2)], [1.0, -0.9], rng.standard_normal(n_days))
    hardness = 1.0 + np.clip(np.repeat(hard_d, 24)[:n], -0.12, 0.12)

    # outages: random trips plus a planned 36 h shutdown roughly every 90 days
    down = np.zeros(n, dtype=bool)
    for _ in range(rng.poisson(n / 240.0)):
        start = int(rng.integers(0, n))
        dur = int(np.clip(rng.lognormal(1.6, 0.6), 1, 30))
        down[start : start + dur] = True
    for day0 in range(45, n_days, 90):
        s = day0 * 24 + 6
        down[s : s + 36] = True

    plant = 11.0 * hardness * np.where(down, 0.08, 1.0)
    shift = np.where((hour >= 5) & (hour < 21), 1.0, 0.55)
    crushing = 2.2 * shift * np.where(dow == 6, 0.9, 1.0) * np.where(down, 0.10, 1.0)
    dewater = (1.2 + 1.3 * wet) * (0.9 + 0.1 * np.sin(2 * np.pi * hour / 12.0))
    camp = (
        0.9
        + 0.5 * np.exp(-0.5 * ((hour - 20.0) / 2.5) ** 2)
        + 0.08 * np.maximum(np.asarray(temp_c) - 24.0, 0.0)
    )
    load = (plant + crushing + dewater + camp) * (1.0 + 0.015 * _ar1(n, 0.8, rng))
    return pd.Series(load, index=index, name="load_mw")
