#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import PhaseSeries


class TestPhaseSeries(object):

    @pytest.mark.parametrize("country", ["Japan"])
    def test_start(self, jhu_data, population_data, country):
        # Dataset
        population = population_data.value(country)
        # Setting
        series = PhaseSeries("01Apr2020", "22Apr2020", population, name="Main")
        assert str(series) == "Main scenario"
