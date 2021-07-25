#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import SubsetNotFoundError, PopulationPyramidData, Term


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
        assert set(df.columns) == set([PopulationPyramidData.AGE, Term.N, PopulationPyramidData.PORTION])
