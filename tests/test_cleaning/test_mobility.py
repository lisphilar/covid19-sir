#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import MobilityData


class TestOxCGRTData(object):
    def test_cleaning(self, mobility_data):
        df = mobility_data.cleaned()
        assert set(df.columns) == set(MobilityData.RAW_COLS)

    def test_subset(self, mobility_data):
        with pytest.raises(KeyError):
            mobility_data.subset("Moon")
        df = mobility_data.subset("JPN")
        assert set(df.columns) == set(MobilityData.SUBSET_COLS)

    def test_total(self, mobility_data):
        with pytest.raises(NotImplementedError):
            mobility_data.total()

    def test_map(self, mobility_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        mobility_data.map(country=None)
        mobility_data.map(country="Japan")
