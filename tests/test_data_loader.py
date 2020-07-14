#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import pytest
import warnings
from covsirphy import DataLoader
from covsirphy import Term, JHUData, CountryData, PopulationData, OxCGRTData


class TestJHUData(object):
    def test_jhu(self, data_loader):
        jhu_data = data_loader.jhu()
        assert isinstance(jhu_data, JHUData)
        assert isinstance(jhu_data.citation, str)
        df = jhu_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Term.COLUMNS)

    def test_jhu_local_file(self, data_loader):
        local_path = Path("input") / "covid19dh.csv"
        data_loader.jhu(local_file=local_path)
        local_file = str(local_path)
        jhu_data = data_loader.jhu(local_file=local_file)
        assert jhu_data.citation == str()

    def test_jhu_local_file_unexpected(self, data_loader):
        local_path = Path("input") / "covid_jpn_total.csv"
        with pytest.raises(Exception):
            data_loader.jhu(local_file=local_path)

    def test_replace(self, data_loader):
        jhu_data = data_loader.jhu()
        japan_data = data_loader.japan()
        jhu_data.replace(japan_data)
        assert isinstance(jhu_data, JHUData)
        replaced_df = jhu_data.subset("Japan")
        japan_df = japan_data.cleaned()
        assert set(replaced_df.columns) == set(Term.NLOC_COLUMNS)
        assert len(replaced_df) == len(japan_df)

    def test_area_name(self, jhu_data):
        name_country = jhu_data.area_name("Japan")
        assert name_country == "Japan"
        name_with_province = jhu_data.area_name("Japan", province="Tokyo")
        assert name_with_province == "Japan/Tokyo"

    def test_subset(self, data_loader):
        jhu_data = data_loader.jhu()
        df = jhu_data.subset("Japan")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
        # Province
        if "Tokyo" in jhu_data.cleaned().columns:
            df = jhu_data.subset("Japan", province="Tokyo")
            assert set(df.columns) == set(Term.NLOC_COLUMNS)
        # Date
        df = jhu_data.subset(
            "Japan", start_date="01Feb2020", end_date="01Mar2020"
        )
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
        # Calculate Susceptible with population value
        df = jhu_data.subset("Japan", population=125000000)
        assert set(df.columns) == set([*Term.NLOC_COLUMNS, Term.S])


class TestJapanData(object):
    def test_japan_cases(self, data_loader):
        japan_data = data_loader.japan()
        assert isinstance(japan_data, CountryData)
        assert isinstance(japan_data.citation, str)
        df = japan_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Term.COLUMNS)

    def test_japan_cases_local_file(self, data_loader):
        local_path = Path("input") / "covid_jpn_total.csv"
        data_loader.japan(local_file=local_path)
        local_file = str(local_path)
        japan_data = data_loader.japan(local_file=local_file)
        assert japan_data.citation == str()

    def test_japan_cases_local_file_unexpected(self):
        data_loader = DataLoader("input")
        local_path = Path("input") / "covid19dh.csv"
        with pytest.raises(Exception):
            warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
            data_loader.japan(local_file=local_path)


class TestPopulationData(object):
    def test_population(self, data_loader):
        population_data = data_loader.population()
        assert isinstance(population_data, PopulationData)
        assert isinstance(population_data.citation, str)
        df = population_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(PopulationData.POPULATION_COLS)
        assert isinstance(population_data.to_dict(), dict)
        population_data.update(10_000, "Example")
        assert population_data.value("Example") == 10_000
        population_data.update(15_000, "Example")
        assert population_data.value("Example") == 15_000

    def test_population_local_file(self, data_loader):
        local_path = Path("input") / "covid19dh.csv"
        data_loader.population(local_file=local_path)
        local_file = str(local_path)
        population_data = data_loader.population(local_file=local_file)
        assert population_data.citation == str()

    def test_population_local_file_unexpected(self, data_loader):
        local_path = Path("input") / "covid_jpn_total.csv"
        with pytest.raises(Exception):
            data_loader.population(local_file=local_path)

    def test_population_value(self, data_loader):
        population_data = data_loader.population()
        df = population_data.cleaned()
        if "JPN" in df["ISO3"].unique():
            assert isinstance(population_data.value("JPN"), int)
        else:
            with pytest.raises(KeyError):
                population_data.value("JPN")
        assert isinstance(population_data.value("Japan"), int)


class TestOxCGRTData(object):
    def test_oxcgrt(self, data_loader):
        oxcgrt_data = data_loader.oxcgrt()
        assert isinstance(oxcgrt_data, OxCGRTData)
        assert isinstance(oxcgrt_data.citation, str)
        df = oxcgrt_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(OxCGRTData.OXCGRT_COLS)
        subset_cols_set = set(OxCGRTData.OXCGRT_COLS_WITHOUT_COUNTRY)
        subset_df = oxcgrt_data.subset(country="Japan")
        assert set(subset_df.columns) == subset_cols_set
        subset_df_iso = oxcgrt_data.subset(iso3="JPN")
        assert set(subset_df_iso.columns) == subset_cols_set

    def test_oxcgrt_local_file(self, data_loader):
        local_path = Path("input") / "covid19dh.csv"
        data_loader.oxcgrt(local_file=local_path)
        local_file = str(local_path)
        oxcgrt_data = data_loader.oxcgrt(local_file=local_file)
        assert oxcgrt_data.citation == str()

    def test_oxcgrt_local_file_unexpected(self, data_loader):
        local_path = Path("input") / "covid_jpn_total.csv"
        with pytest.raises(Exception):
            data_loader.oxcgrt(local_file=local_path)
