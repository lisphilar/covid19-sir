#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import pytest
import warnings
from covsirphy import DataLoader
from covsirphy import Word, JHUData, CountryData, PopulationData, OxCGRTData


class TestDataLoader(object):
    def test_jhu(self, data_loader):
        jhu_data = data_loader.jhu()
        assert isinstance(jhu_data, JHUData)
        assert isinstance(jhu_data.citation, str)
        df = jhu_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.COLUMNS)

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

    def test_japan_cases(self, data_loader):
        japan_data = data_loader.japan()
        assert isinstance(japan_data, CountryData)
        assert isinstance(japan_data.citation, str)
        df = japan_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.COLUMNS)

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
            warnings.filterwarnings("ignore", pd.errors.DtypeWarning)
            data_loader.japan(local_file=local_path)

    def test_subset(self, data_loader):
        jhu_data = data_loader.jhu()
        df = jhu_data.subset("Japan")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.NLOC_COLUMNS)

    def test_replace(self, data_loader):
        jhu_data = data_loader.jhu()
        japan_data = data_loader.japan()
        jhu_data.replace(japan_data)
        assert isinstance(jhu_data, JHUData)
        replaced_df = jhu_data.subset("Japan")
        japan_df = japan_data.cleaned()
        assert set(replaced_df.columns) == set(Word.NLOC_COLUMNS)
        assert len(replaced_df) == len(japan_df)

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
