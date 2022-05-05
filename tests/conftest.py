#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import pandas as pd
import pytest
from covsirphy import DataLoader, JapanData


@pytest.fixture(scope="session")
def data_loader():
    return DataLoader()


@pytest.fixture(scope="session")
def jhu_data(data_loader):
    return data_loader.jhu()


@pytest.fixture(scope="session")
def population_data(data_loader):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    return data_loader.population()


@pytest.fixture(scope="session")
def oxcgrt_data(data_loader):
    return data_loader.oxcgrt()


@pytest.fixture(scope="session")
def japan_data(data_loader):
    return data_loader.japan()


@pytest.fixture(scope="session")
def pcr_data(data_loader):
    return data_loader.pcr()


@pytest.fixture(scope="session")
def vaccine_data(data_loader):
    return data_loader.vaccine()


@pytest.fixture(scope="session")
def mobility_data(data_loader):
    return data_loader.mobility()


@pytest.fixture(scope="session")
def pyramid_data(data_loader):
    return data_loader.pyramid()


@pytest.fixture(
    scope="session",
    params=[
        "jhu_data", "population_data", "oxcgrt_data", "japan_data",
        "pcr_data", "vaccine_data", "mobility_data", "pyramid_data",
    ])
def data(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def imgfile():
    dirpath = Path("input")
    dirpath.mkdir(exist_ok=True)
    filepath = dirpath.joinpath("test.jpg")
    yield str(filepath)
    try:
        filepath.unlink(missing_ok=True)
    except TypeError:
        # Python 3.7
        if filepath.exists():
            filepath.unlink()


@pytest.fixture(scope="session")
def c_df():
    path = Path("data", "japan", "covid_jpn_total.csv")
    filepath = path if path.exists() else JapanData.URL_C
    df = pd.read_csv(filepath, dayfirst=False).rename(columns={"Date": "date"})
    df = df[["date", "Positive", "Tested", "Discharged", "Fatal"]].groupby("date", as_index=False).sum()
    df.insert(0, "Country", "Japan")
    return df


@pytest.fixture(scope="session")
def p_df():
    path = Path("data", "japan", "covid_jpn_prefecture.csv")
    filepath = path if path.exists() else JapanData.URL_P
    df = pd.read_csv(filepath, dayfirst=False).rename(columns={"Date": "date"})
    df.insert(0, "Country", "Japan")
    return df[["Country", "Prefecture", "date", "Positive", "Tested", "Discharged", "Fatal"]]


@pytest.fixture(scope="session")
def japan_df(c_df, p_df):
    df = pd.concat([c_df, p_df], axis=0)
    assert set(df.columns) == {"Country", "Prefecture", "date", "Positive", "Tested", "Discharged", "Fatal"}
    return df
