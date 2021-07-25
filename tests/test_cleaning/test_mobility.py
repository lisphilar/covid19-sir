#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import Term


class TestMobilityData(object):
    def test_cleaning(self, mobility_data):
        df = mobility_data.cleaned()
        assert set([Term.DATE, Term.ISO3, Term.COUNTRY, Term.PROVINCE]).issubset(df.columns)

    def test_subset(self, mobility_data):
        with pytest.raises(KeyError):
            mobility_data.subset("Moon")
        df = mobility_data.subset("JPN")
        assert set([Term.DATE]).issubset(df.columns)

    def test_total(self, mobility_data):
        with pytest.raises(NotImplementedError):
            mobility_data.total()

    def test_map(self, mobility_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        mobility_data.map(country=None)
        mobility_data.map(country="Japan")
