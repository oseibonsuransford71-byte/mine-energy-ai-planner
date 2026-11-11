"""Leakage tests: a forecast issued at time t must not depend on anything observed after t."""
import numpy as np
import pandas as pd
import pytest

from minegrid.data import build_dataset
from minegrid.forecast import DETERMINISTIC_COLS, _split, load_features, solar_features


@pytest.fixture(scope="module")
def df():
    return build_dataset("2022-01-01 00:00", "2022-03-31 23:00")


def corrupt_future(df, issue_time):
    bad = df.copy()
    rng = np.random.default_rng(0)
    mask = bad.index > issue_time
    for col in ("kcs", "temp", "load_mw", "ghi", "pv_cf"):
        bad.loc[mask, col] = rng.uniform(-500, 500, mask.sum())
    return bad


@pytest.mark.parametrize("h", [1, 6, 24])
def test_solar_features_use_only_data_up_to_issue_time(df, h):
    target = df.index[1000]
    issue = target - pd.Timedelta(hours=h)
    clean = solar_features(df, h).loc[target]
    dirty = solar_features(corrupt_future(df, issue), h).loc[target]
    cols = [c for c in clean.index if c not in DETERMINISTIC_COLS]
    assert np.allclose(clean[cols].to_numpy(float), dirty[cols].to_numpy(float))


def test_load_features_use_only_data_up_to_issue_time(df):
    target = df.index[1000]
    issue = target - pd.Timedelta(hours=24)
    clean = load_features(df, 24).loc[target]
    dirty = load_features(corrupt_future(df, issue), 24).loc[target]
    cols = [c for c in clean.index if c not in DETERMINISTIC_COLS]
    assert np.allclose(clean[cols].to_numpy(float), dirty[cols].to_numpy(float))


def test_train_test_split_is_chronological():
    idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h")
    train, test = _split(idx)
    assert idx[train].max() < idx[test].min()
    assert idx[test].min().year == 2023
    assert not np.any(train & test)


def test_horizon_longer_than_a_day_is_rejected(df):
    with pytest.raises(AssertionError):
        solar_features(df, 25)
