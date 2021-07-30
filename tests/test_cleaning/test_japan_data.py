#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
import pandas as pd
from covsirphy import Term, JapanData


class TestJapanData(object):
    def test_cleaning(self, japan_data):
        assert isinstance(japan_data.raw_columns(), list)
        df = japan_data.cleaned()
        assert set(Term.COLUMNS).issubset(df.columns)

    def test_total(self, japan_data):
        japan_data.register_total()
        df = japan_data.total()
        column_set = set(Term.VALUE_COLUMNS) | set(Term.RATE_COLUMNS)
        assert set(df.columns) == column_set

    def test_countries(self, japan_data):
        assert [japan_data.country] == japan_data.countries()

    def test_japan_meta(self, japan_data):
        raw_df = japan_data.meta(cleaned=False)
        assert isinstance(raw_df, pd.DataFrame)
        df = japan_data.meta(cleaned=True)
        assert set(df.columns) == set(JapanData.JAPAN_META_COLS)

    def test_map(self, japan_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        japan_data.map()
        with pytest.raises(NotImplementedError):
            japan_data.map(country="GBR")
