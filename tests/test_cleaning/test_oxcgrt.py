#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import OxCGRTData


class TestOxCGRTData(object):
    def test_cleaning(self, oxcgrt_data):
        df = oxcgrt_data.cleaned()
        assert set(df.columns) == set(OxCGRTData.OXCGRT_COLS)

    def test_subset(self, oxcgrt_data):
        with pytest.raises(KeyError):
            oxcgrt_data.subset("Moon")
        df = oxcgrt_data.subset("JPN")
        assert set(df.columns) == set(OxCGRTData.OXCGRT_COLS_WITHOUT_COUNTRY)

    def test_total(self, oxcgrt_data):
        with pytest.raises(NotImplementedError):
            oxcgrt_data.total()

    def test_map(self, oxcgrt_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        oxcgrt_data.map(country=None)
        with pytest.raises(NotImplementedError):
            oxcgrt_data.map(country="Japan")
