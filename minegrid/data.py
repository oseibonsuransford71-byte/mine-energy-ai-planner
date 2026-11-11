"""Dataset construction: simulated by default, real NASA POWER irradiance optional."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_SITE, END, SEED, START, Site
from .solar import clear_sky_ghi, pv_specific_power, solar_geometry
from .synth import harmattan, synth_load, synth_weather


def soiling_factor(index: pd.DatetimeIndex) -> np.ndarray:
    """Dust soiling loss: worse during the harmattan season (Dec-Feb)."""
    return 1.0 - 0.02 - 0.04 * harmattan(np.asarray(index.dayofyear))


def weather_from_nasa_power(payload: dict) -> pd.DataFrame:
    """Parse a NASA POWER hourly JSON payload (parameters ALLSKY_SFC_SW_DWN and T2M).

    Returns a DataFrame with columns ghi (W/m2) and temp (degC) on an hourly index.
    Fill values (-999) are treated as missing and interpolated over gaps of up to 6 h.
    """
    params = payload["properties"]["parameter"]
    ghi = pd.Series(params["ALLSKY_SFC_SW_DWN"], dtype=float)
    temp = pd.Series(params["T2M"], dtype=float)
    idx = pd.to_datetime(ghi.index, format="%Y%m%d%H")
    out = pd.DataFrame({"ghi": ghi.to_numpy(), "temp": temp.reindex(ghi.index).to_numpy()}, index=idx)
    out = out.sort_index().replace(-999.0, np.nan)
    out = out.interpolate(limit=6, limit_direction="both")
    out["ghi"] = out["ghi"].clip(lower=0.0)
    return out.dropna()


def read_weather_file(path: str | Path) -> pd.DataFrame:
    """Read either a NASA POWER JSON file or a CSV with columns time, ghi, temp."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return weather_from_nasa_power(json.loads(path.read_text()))
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df[["ghi", "temp"]]


def derive_columns(weather: pd.DataFrame, site: Site = DEFAULT_SITE) -> pd.DataFrame:
    """Add clear-sky irradiance, clear-sky index and PV specific power to a weather frame."""
    df = weather.copy()
    cosz, _ = solar_geometry(df.index, site.lat, site.lon, site.tz_hours)
    df["cosz"] = cosz
    df["ghi_cs"] = clear_sky_ghi(cosz)
    kcs = np.where(df["ghi_cs"] > 20.0, df["ghi"] / np.maximum(df["ghi_cs"], 1e-6), np.nan)
    kcs = pd.Series(kcs, index=df.index).clip(0.0, 1.2)
    # night-time clear-sky index is undefined: carry the last daylight value forward
    df["kcs"] = kcs.ffill().bfill()
    df["pv_cf"] = pv_specific_power(df["ghi"], df["temp"], soiling_factor(df.index))
    return df


def build_dataset(
    start: str = START,
    end: str = END,
    seed: int = SEED,
    site: Site = DEFAULT_SITE,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Hourly dataset: ghi, temp, cosz, ghi_cs, kcs, pv_cf, load_mw."""
    if weather is None:
        index = pd.date_range(start, end, freq="h")
        weather = synth_weather(index, site, seed)
    df = derive_columns(weather, site)
    df["load_mw"] = synth_load(df.index, df["temp"].to_numpy(), seed + 1).to_numpy()
    return df
