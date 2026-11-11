"""Solar geometry, clear-sky irradiance and a simple PV power model.

References
----------
- Spencer (1971) Fourier series for declination, equation of time and eccentricity.
- Haurwitz (1945) clear-sky global horizontal irradiance model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G_SC = 1361.0  # W/m2 solar constant


def solar_geometry(index: pd.DatetimeIndex, lat_deg: float, lon_deg: float, tz_hours: float = 0.0):
    """Return (cos_zenith, eccentricity_correction) for hour-beginning timestamps.

    Timestamps are local standard time. Geometry is evaluated at mid-hour so that it matches
    hour-averaged irradiance.
    """
    ts = index + pd.Timedelta(minutes=30)
    doy = np.asarray(ts.dayofyear, dtype=float)
    hour = np.asarray(ts.hour, dtype=float) + np.asarray(ts.minute, dtype=float) / 60.0
    b = 2.0 * np.pi * (doy - 1.0) / 365.0

    decl = (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2 * b)
        + 0.000907 * np.sin(2 * b)
        - 0.002697 * np.cos(3 * b)
        + 0.00148 * np.sin(3 * b)
    )
    eot_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(b)
        - 0.032077 * np.sin(b)
        - 0.014615 * np.cos(2 * b)
        - 0.04089 * np.sin(2 * b)
    )
    e0 = (
        1.000110
        + 0.034221 * np.cos(b)
        + 0.001280 * np.sin(b)
        + 0.000719 * np.cos(2 * b)
        + 0.000077 * np.sin(2 * b)
    )
    solar_time = hour + (4.0 * (lon_deg - 15.0 * tz_hours) + eot_min) / 60.0
    omega = np.radians(15.0 * (solar_time - 12.0))
    lat = np.radians(lat_deg)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(omega)
    return cosz, e0


def clear_sky_ghi(cosz: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky GHI in W/m2."""
    cosz = np.asarray(cosz, dtype=float)
    safe = np.maximum(cosz, 1e-3)
    return np.where(cosz > 0.0, 1098.0 * cosz * np.exp(-0.057 / safe), 0.0)


def pv_specific_power(
    ghi,
    temp_c,
    soiling=1.0,
    gamma: float = -0.0035,
    derate: float = 0.88,
    dc_ac: float = 1.2,
    noct: float = 45.0,
) -> np.ndarray:
    """AC output per unit DC capacity (kW per kWp, equal to MW per MWp).

    Simple single-point model: cell temperature from NOCT, linear temperature coefficient,
    a lumped derate (inverter, wiring, mismatch), a soiling factor and inverter clipping at
    1 / dc_ac of the DC rating.
    """
    ghi = np.asarray(ghi, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    t_cell = temp_c + (noct - 20.0) / 800.0 * ghi
    p = ghi / 1000.0 * (1.0 + gamma * (t_cell - 25.0)) * derate * np.asarray(soiling, dtype=float)
    return np.clip(p, 0.0, 1.0 / dc_ac)
