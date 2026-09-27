from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest


def month_ends(n: int, start: str = "1980-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def business_days(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _alphafade_warnings_are_errors() -> Iterator[None]:
    """Fail on any alphafade warning a test doesn't explicitly expect with pytest.warns.

    Done here rather than in pyproject's filterwarnings, which would import alphafade before
    coverage starts measuring.
    """
    from alphafade import AlphaFadeWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", AlphaFadeWarning)
        yield
