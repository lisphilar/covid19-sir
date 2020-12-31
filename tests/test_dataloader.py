#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.japan_data import JapanData
from covsirphy.util.error import SubsetNotFoundError
import pytest
import warnings
import pandas as pd
from covsirphy import CleaningBase, SIRF
from covsirphy import LinelistData, ExampleData, VaccineData
from covsirphy import Term, CountryData, PopulationPyramidData
from covsirphy import Word, Population


class TestLinelistData(object):

    def test_raw(self, linelist_data):
        assert isinstance(linelist_data.raw, pd.DataFrame)

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

    def test_japan_meta(self, japan_data):
        raw_df = japan_data.meta(cleaned=False)
        assert isinstance(raw_df, pd.DataFrame)
        df = japan_data.meta(cleaned=True)
        assert set(df.columns) == set(JapanData.JAPAN_META_COLS)


class TestVaccineData(object):
    def test_cleaning(self, vaccine_data):
        df = vaccine_data.cleaned()
        assert set(VaccineData.VAC_COLS).issubset(df.columns)

    @pytest.mark.parametrize("country", ["Canada"])
    def test_subset(self, vaccine_data, country):
        df = vaccine_data.subset(country=country)
        assert set(df.columns) == set([Term.DATE, Term.VAC])
        clean_df = vaccine_data.cleaned()
        df = clean_df.loc[clean_df[Term.COUNTRY] == country, :]
        product = df.loc[df.index[0], Term.PRODUCT]
        vaccine_data.subset(
            country=country, product=product, start_date="15Dec2020", end_date="18Dec2020")
        with pytest.raises(SubsetNotFoundError):
            vaccine_data.subset(country=country, end_date="01May2020")

    @pytest.mark.parametrize("country", ["GBR"])
    def test_records(self, vaccine_data, country):
        vaccine_data.records(country=country)

    def test_total(self, vaccine_data):
        df = vaccine_data.total()
        assert set(df.columns) == set([Term.DATE, Term.VAC])


class TestPopulationPyramidData(object):
    def test_retrieve(self, pyramid_data):
        df = pyramid_data.retrieve("Japan")
        assert set(df.columns) == set(PopulationPyramidData.PYRAMID_COLS)
        with pytest.raises(SubsetNotFoundError):
            pyramid_data.retrieve("Moon")

    def test_cleaned(self, pyramid_data):
        df = pyramid_data.cleaned()
        assert set(df.columns) == set(PopulationPyramidData.PYRAMID_COLS)

    @pytest.mark.parametrize("country", ["Japan"])
    @pytest.mark.parametrize("sex", [None, "Female", "Male"])
    def test_records(self, pyramid_data, country, sex):
        df = pyramid_data.records(country, sex=sex)
        assert set(df.columns) == set(PopulationPyramidData.SUBSET_COLS)
