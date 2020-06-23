#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy import Word
from covsirphy import DataLoader, JHUData, CountryData


class TestDataLoader(object):
    def test_jhu(self):
        data_loader = DataLoader("input")
        jhu_data = data_loader.jhu()
        assert isinstance(jhu_data, JHUData)
        assert isinstance(jhu_data.citation, str)
        df = jhu_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.COLUMNS)

    def test_japan_cases(self):
        data_loader = DataLoader("input")
        japan_data = data_loader.japan()
        assert isinstance(japan_data, CountryData)
        assert isinstance(japan_data.citation, str)
        df = japan_data.cleaned()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.COLUMNS)

    def test_subset(self):
        data_loader = DataLoader("input")
        jhu_data = data_loader.jhu()
        df = jhu_data.subset("Japan")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Word.COLUMNS)

    def test_replace(self):
        data_loader = DataLoader("input")
        jhu_data = data_loader.jhu()
        japan_data = data_loader.japan()
        jhu_data.replace(japan_data)
        assert isinstance(jhu_data, JHUData)
        replaced_df = jhu_data.subset("Japan")
        japan_df = japan_data.cleaned()
        assert set(replaced_df.columns) == set(japan_df.columns)
        assert len(replaced_df) == len(japan_df)
