#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
import pandas as pd
from covsirphy import CleaningBase, SIRF
from covsirphy import LinelistData, ExampleData
from covsirphy import Term, CountryData
from covsirphy import Word, Population


class TestLinelistData(object):
    def test_linelist(self, linelist_data):
        with pytest.raises(NotImplementedError):
            linelist_data.total()
        assert isinstance(linelist_data.cleaned(), pd.DataFrame)
        assert isinstance(linelist_data.citation, str)

    @pytest.mark.parametrize("country", ["Japan", "Germany"])
    @pytest.mark.parametrize("province", [None, "Tokyo"])
    def test_subset(self, linelist_data, country, province):
        if (country, province) == ("Germany", "Tokyo"):
            with pytest.raises(KeyError):
                linelist_data.subset(country=country, province=province)
        else:
            df = linelist_data.subset(country=country, province=province)
            column_set = set(df) | set([Term.COUNTRY, Term.PROVINCE])
            assert column_set == set(LinelistData.LINELIST_COLS)

    @pytest.mark.parametrize("outcome", ["Recovered", "Fatal", "Confirmed"])
    def test_closed(self, linelist_data, outcome):
        if outcome in ["Recovered", "Fatal"]:
            linelist_data.closed(outcome=outcome)
        else:
            with pytest.raises(KeyError):
                linelist_data.closed(outcome=outcome)

    def test_recovery_period(self, linelist_data):
        assert isinstance(linelist_data.recovery_period(), int)


class TestObsoleted(object):
    def test_obsoleted(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        Population(filename=None)
        Word()


class TestCleaningBase(object):
    def test_cbase(self):
        cbase = CleaningBase(filename=None)
        with pytest.raises(KeyError):
            cbase.iso3_to_country("JPN")
        with pytest.raises(NotImplementedError):
            cbase.total()
        cbase.citation = "citation"
        assert cbase.citation == "citation"


class TestExampleData(object):
    def test_iso3(self):
        example_data = ExampleData()
        example_data.add(SIRF, country="Japan")
        assert example_data.country_to_iso3("Japan") == "JPN"
        example_data.add(SIRF, country="Moon")
        assert example_data.country_to_iso3("Moon") == "---"

    def test_subset(self):
        example_data = ExampleData()
        example_data.add(SIRF, country="Japan")
        example_data.subset(country="Japan")
        example_data.subset_complement(country="Japan")
        example_data.records(country="Japan")


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
