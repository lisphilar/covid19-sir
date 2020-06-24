#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import pytest
from covsirphy import Word
from covsirphy import DataLoader, JHUData, CountryData, OxCGRTData


class TestDataLoader(object):
    def test_jhu(self):
        data_loader = DataLoader("input")
        jhu_data = data_loader.jhu()
        assert isinstance(jhu_data, JHUData)
        assert isinstance(jhu_data.citation, str)
        df = jhu_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.COLUMNS)

    def test_jhu_local_file(self):
        data_loader = DataLoader("input")
        local_path = Path("input") / "covid_19_data.csv"
        data_loader.jhu(local_file=local_path)
        local_file = str(local_path)
        data_loader.jhu(local_file=local_file)

    def test_jhu_local_file_unexpected(self):
        data_loader = DataLoader("input")
        local_path = Path("input") / "covid_jpn_total.csv"
        with pytest.raises(KeyError):
            data_loader.jhu(local_file=local_path)

    def test_japan_cases(self):
        data_loader = DataLoader("input")
        japan_data = data_loader.japan()
        assert isinstance(japan_data, CountryData)
        assert isinstance(japan_data.citation, str)
        df = japan_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.COLUMNS)

    def test_japan_cases_local_file(self):
        data_loader = DataLoader("input")
        local_path = Path("input") / "covid_jpn_total.csv"
        data_loader.japan(local_file=local_path)
        local_file = str(local_path)
        data_loader.japan(local_file=local_file)

    def test_japan_cases_local_file_unexpected(self):
        data_loader = DataLoader("input")
        local_path = Path("input") / "covid_19_data.csv"
        with pytest.raises(Exception):
            data_loader.japan(local_file=local_path)

    def test_subset(self):
        data_loader = DataLoader("input")
        jhu_data = data_loader.jhu()
        df = jhu_data.subset("Japan")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.NLOC_COLUMNS)

    def test_replace(self):
        data_loader = DataLoader("input")
        jhu_data = data_loader.jhu()
        japan_data = data_loader.japan()
        jhu_data.replace(japan_data)
        assert isinstance(jhu_data, JHUData)
        replaced_df = jhu_data.subset("Japan")
        japan_df = japan_data.cleaned()
        assert set(replaced_df.columns) == set(Word.NLOC_COLUMNS)
        assert len(replaced_df) == len(japan_df)

    def test_oxcgrt(self):
        data_loader = DataLoader("input")
        oxcgrt_data = data_loader.oxcgrt()
        assert isinstance(oxcgrt_data, OxCGRTData)
        assert isinstance(oxcgrt_data.citation, str)
        df = oxcgrt_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(OxCGRTData.OXCGRT_COLS)

    def test_oxcgrt_local_file(self):
        data_loader = DataLoader("input")
        local_path = Path("input") / "OxCGRT_latest.csv"
        data_loader.oxcgrt(local_file=local_path)
        local_file = str(local_path)
        data_loader.oxcgrt(local_file=local_file)

    def test_oxcgrt_local_file_unexpected(self):
        data_loader = DataLoader("input")
        local_path = Path("input") / "covid_jpn_total.csv"
        with pytest.raises(Exception):
            data_loader.oxcgrt(local_file=local_path)
