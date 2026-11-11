#!/usr/bin/env python3
"""Download real hourly irradiance and temperature from NASA POWER and save a weather CSV.

    python scripts/fetch_nasa_power.py                      # default site, 2021-2023
    python scripts/fetch_nasa_power.py --lat 6.2 --lon -1.67 --start 2021 --end 2023
    python scripts/run_pipeline.py --weather data/raw/nasa_power_weather.csv

Then the whole pipeline (forecasting, sizing, forecast value) reruns on measured-quality weather.
Mine load is still simulated because real mine load data is confidential.

Notes
-----
* NASA POWER is free and needs no API key: https://power.larc.nasa.gov/docs/services/api/
* Parameters: ALLSKY_SFC_SW_DWN (all-sky GHI, hourly) and T2M (2 m air temperature).
* The parsing code (minegrid.data.weather_from_nasa_power) is unit-tested on a payload with the
  documented structure. The live request itself was written against the API documentation and
  could not be exercised from the sandbox this project was built in, so run it once and check
  the printed summary before trusting the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from minegrid.data import weather_from_nasa_power  # noqa: E402

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_year(lat: float, lon: float, year: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": f"{year}0101",
            "end": f"{year}1231",
            "format": "JSON",
            "time-standard": "UTC",
        }
    )
    with urllib.request.urlopen(f"{API}?{query}", timeout=120) as resp:
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=6.20)
    ap.add_argument("--lon", type=float, default=-1.67)
    ap.add_argument("--start", type=int, default=2021, help="first year")
    ap.add_argument("--end", type=int, default=2023, help="last year")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "nasa_power_weather.csv"))
    args = ap.parse_args()

    frames = []
    for year in range(args.start, args.end + 1):
        print(f"fetching {year} ...", flush=True)
        frames.append(weather_from_nasa_power(fetch_year(args.lat, args.lon, year)))
    weather = pd.concat(frames).sort_index()
    weather = weather[~weather.index.duplicated()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(args.out, index_label="time")
    print(f"saved {len(weather)} hourly rows to {args.out}")
    print(f"mean GHI {weather['ghi'].resample('D').sum().mean() / 1000:.2f} kWh/m2/day, mean temp {weather['temp'].mean():.1f} C")


if __name__ == "__main__":
    main()
