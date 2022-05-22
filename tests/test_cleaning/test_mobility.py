#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import Term, SubsetNotFoundError


class TestMobilityData(object):
    def test_cleaning(self, mobility_data):
        df = mobility_data.cleaned()
        assert {Term.DATE, Term.ISO3, Term.COUNTRY, Term.PROVINCE}.issubset(df.columns)

    def test_subset(self, mobility_data):
        with pytest.raises(SubsetNotFoundError):
            mobility_data.subset("Moon")
        df = mobility_data.subset("JPN")
        assert {Term.DATE}.issubset(df.columns)

    def test_total(self, mobility_data):
        with pytest.raises(NotImplementedError):
            mobility_data.total()

    def test_map(self, mobility_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        mobility_data.map(country=None)
        mobility_data.map(country="Japan")
