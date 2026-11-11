"""Project-wide constants. Everything an assessor might want to change lives here or in the dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

START = "2021-01-01 00:00"
END = "2023-12-31 23:00"
TRAIN_END = "2022-12-31 23:00"  # train on 2021-2022, test on 2023 (strict time split)
SEED = 42


@dataclass(frozen=True)
class Site:
    """An illustrative site. NOT a real mine: coordinates only set the solar geometry and climate."""

    name: str = "Illustrative gold mine near Obuasi, Ashanti Region, Ghana"
    lat: float = 6.20
    lon: float = -1.67
    tz_hours: float = 0.0  # Ghana is on UTC+0 all year
    diesel_capacity_mw: float = 24.0  # installed gensets (N+1), must cover the peak load


DEFAULT_SITE = Site()
