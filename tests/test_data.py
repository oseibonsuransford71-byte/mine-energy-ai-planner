import numpy as np
import pandas as pd

from minegrid.data import build_dataset, read_weather_file, weather_from_nasa_power


def fake_nasa_payload():
    """Same structure as a NASA POWER hourly point response (keys are YYYYMMDDHH)."""
    keys = [f"202301{d:02d}{h:02d}" for d in (1, 2) for h in range(24)]
    ghi = {k: float(max(0, 800 * np.sin(np.pi * (int(k[-2:]) - 6) / 12))) for k in keys}
    ghi[keys[10]] = -999.0  # fill value that must be interpolated
    temp = {k: 26.0 + 4 * np.sin(np.pi * (int(k[-2:]) - 9) / 12) for k in keys}
    return {"properties": {"parameter": {"ALLSKY_SFC_SW_DWN": ghi, "T2M": temp}}}


def test_nasa_power_parser_handles_fill_values():
    w = weather_from_nasa_power(fake_nasa_payload())
    assert len(w) == 48
    assert isinstance(w.index, pd.DatetimeIndex) and w.index.is_monotonic_increasing
    assert not w.isna().any().any()
    assert (w["ghi"] >= 0).all()
    assert w["ghi"].iloc[10] > 0  # the -999 hour was interpolated, not kept


def test_real_weather_can_drive_the_full_dataset(tmp_path):
    w = weather_from_nasa_power(fake_nasa_payload())
    path = tmp_path / "w.csv"
    w.to_csv(path, index_label="time")
    df = build_dataset(weather=read_weather_file(path))
    assert {"ghi", "temp", "kcs", "pv_cf", "load_mw", "ghi_cs"} <= set(df.columns)
    assert len(df) == 48 and not df.isna().any().any()


def test_dataset_is_reproducible():
    a = build_dataset("2022-01-01 00:00", "2022-01-10 23:00", seed=7)
    b = build_dataset("2022-01-01 00:00", "2022-01-10 23:00", seed=7)
    c = build_dataset("2022-01-01 00:00", "2022-01-10 23:00", seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["ghi"], c["ghi"])
