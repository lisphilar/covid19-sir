#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import pandas as pd
from covsirphy import GIS, JapanData, NotRegisteredError, SubsetNotFoundError


@pytest.fixture(scope="module")
def c_df():
    path = Path("data", "japan", "covid_jpn_total.csv")
    filepath = path if path.exists() else JapanData.URL_C
    df = pd.read_csv(filepath, dayfirst=False).rename(columns={"Date": "date"})
    df = df[["date", "Positive", "Tested"]].groupby("date", as_index=False).sum()
    df.insert(0, "Country", "Japan")
    return df


@pytest.fixture(scope="module")
def p_df():
    path = Path("data", "japan", "covid_jpn_prefecture.csv")
    filepath = path if path.exists() else JapanData.URL_P
    df = pd.read_csv(filepath, dayfirst=False).rename(columns={"Date": "date"})
    df.insert(0, "Country", "Japan")
    return df[["Country", "Prefecture", "date", "Positive", "Tested"]]


class TestGIS(object):
    def test_all(self, c_df, p_df):
        system = GIS(layers=["Country", "Province"], country="Country", date="Date")
        with pytest.raises(NotRegisteredError):
            system.all(variables=["Positive"])
        system.register(data=c_df, layers=["Country"], date="date", citations="Country-level")
        system.register(data=p_df, layers=["Country", "Prefecture"], date="date", citations="Prefecture-level")
        all_df = system.all(variables=["Positive"])
        assert all_df.columns.tolist() == ["Country", "Province", "Date", "Positive"]
        assert set(system.citations()) == {"Country-level", "Prefecture-level"}

    @pytest.mark.parametrize(
        "geo, end_date, length",
        [
            (None, "31Dec2021", 365),
            ((None,), "31Dec2021", 365),
            ("Japan", "31Dec2020", 0),
            ("Japan", "31Dec2021", 365 * 47),
            (("Japan",), "31Dec2021", 365 * 47),
            ((["Japan", "UK"],), "31Dec2021", 365 * 47),
            ("UK", "31Dec2021", 0),
        ]
    )
    def test_layer(self, c_df, p_df, geo, end_date, length):
        system = GIS(layers=["Country", "Province"], country="Country", date="Date")
        with pytest.raises(NotRegisteredError):
            system.layer(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
        system.register(data=c_df, layers=["Country"], date="date", citations="Country-level")
        system.register(data=p_df, layers=["Country", "Prefecture"], date="date", citations="Prefecture-level")
        if length == 0:
            with pytest.raises(NotRegisteredError):
                system.layer(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
        else:
            df = system.layer(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
            assert df.columns.tolist() == ["Country", "Province", "Date", "Positive", "Tested"]
            assert len(df) == length

    @pytest.mark.parametrize(
        "geo, end_date, length",
        [
            (None, "31Dec2021", 365),
            ((None,), "31Dec2021", 365),
            ("Japan", "31Dec2020", 0),
            ("Japan", "31Dec2021", 365),
            (("Japan",), "31Dec2021", 365),
            ((["Japan", "UK"],), "31Dec2021", 365),
            ("UK", "31Dec2021", 0),
        ]
    )
    def test_subset(self, c_df, p_df, geo, end_date, length):
        system = GIS(layers=["Country", "Province"], country="Country", date="Date")
        with pytest.raises(NotRegisteredError):
            system.subset(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
        system.register(data=c_df, layers=["Country"], date="date", citations="Country-level")
        system.register(data=p_df, layers=["Country", "Prefecture"], date="date", citations="Prefecture-level")
        if length == 0:
            with pytest.raises(SubsetNotFoundError):
                system.subset(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
        else:
            df = system.subset(geo=geo, start_date="01Jan2021", end_date=end_date, variables=None)
            assert df.columns.tolist() == ["Date", "Positive", "Tested"]
            assert len(df) == length
