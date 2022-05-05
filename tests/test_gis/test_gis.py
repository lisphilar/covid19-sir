#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import GIS, NotRegisteredError, SubsetNotFoundError


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
            assert set(df.columns) == {"Country", "Province", "Date", "Positive", "Tested", "Discharged", "Fatal"}
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
            assert set(df.columns) == {"Date", "Positive", "Tested", "Discharged", "Fatal"}
            assert len(df) == length

    @pytest.mark.parametrize(
        "geo, answer",
        [
            (None, "the world"),
            ((None,), "the world"),
            ("Japan", "Japan"),
            (("Japan",), "Japan"),
            ((["Japan", "UK"],), "Japan_UK"),
            (("Japan", "Tokyo"), "Tokyo/Japan"),
        ]
    )
    def test_area_name(self, geo, answer):
        assert GIS.area_name(geo) == answer
