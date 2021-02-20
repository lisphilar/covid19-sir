#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
import covid19dh
import pandas as pd
from covsirphy import Term, CountryData, JapanData


class TestCountryData(object):
    def test_cleaning(self, japan_data):
        assert isinstance(japan_data.raw_columns(), list)
        with pytest.raises(NotImplementedError):
            japan_data.set_variables()
        df = japan_data.cleaned()
        assert set(Term.COLUMNS).issubset(df.columns)

    def test_total(self, japan_data):
        df = japan_data.total()
        column_set = set(Term.VALUE_COLUMNS) | set(Term.RATE_COLUMNS)
        assert set(df.columns) == column_set

    def test_countries(self, japan_data):
        assert [japan_data.country] == japan_data.countries()

    def test_create(self, japan_data):
        country_data = CountryData(filename=None, country="Moon")
        with pytest.raises(ValueError):
            country_data.cleaned()
        country_data.raw = japan_data.raw
        country_data.set_variables(
            date="Date",
            confirmed="Positive",
            fatal="Fatal",
            recovered="Discharged",
            province="Area")
        df = country_data.cleaned()
        assert set(df.columns) == set(Term.COLUMNS)

    def test_create_province(self, japan_data):
        country_data = CountryData(
            filename=None, country="Moon", province="Reiner Gamma")
        country_data.raw = japan_data.raw
        country_data.set_variables(
            date="Date",
            confirmed="Positive",
            fatal="Fatal",
            recovered="Discharged",
            province=None)
        df = country_data.cleaned()
        assert set(df.columns) == set(Term.COLUMNS)

    def test_japan_meta(self, japan_data):
        raw_df = japan_data.meta(cleaned=False)
        assert isinstance(raw_df, pd.DataFrame)
        df = japan_data.meta(cleaned=True)
        assert set(df.columns) == set(JapanData.JAPAN_META_COLS)

    def test_register_total(self):
        # Directly download province level data from COVID-19 Data Hub
        raw_df, *_ = covid19dh.covid19("Italy", level=2, verbose=False)
        filename = "input/italy_raw.csv"
        raw_df.to_csv(filename)
        # Create CountryData instance
        country_data = CountryData(filename=filename, country="Italy")
        country_data.set_variables(
            date="date", confirmed="confirmed", recovered="recovered", fatal="deaths",
            province="administrative_area_level_2"
        )
        # Register total value of all provinces as country level data
        country_data.register_total()
        provinces = country_data.cleaned()[Term.PROVINCE].unique()
        assert Term.UNKNOWN in provinces

    def test_map(self, japan_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        japan_data.map()
        with pytest.raises(NotImplementedError):
            japan_data.map(country="GBR")
