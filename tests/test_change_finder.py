#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import ChangeFinder
from covsirphy import PhaseSeries


class TestChangeFinder(object):
    @pytest.mark.parametrize(
        "country",
        ["Italy", "Japan", "United States", "India", "New Zealand"]
    )
    def test_find(self, jhu_data, population_data, country):
        population = population_data.value(country)
        change_finder = ChangeFinder(
            jhu_data, population, country=country
        )
        change_finder.run()
        phase_series = change_finder.show()
        assert isinstance(phase_series, PhaseSeries)

    def test_find_with_dataframe(self, jhu_data, population_data):
        population = population_data.value("Italy")
        clean_df = jhu_data.cleaned()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        change_finder = ChangeFinder(
            clean_df, population, country="Italy"
        )
        change_finder.run()
        phase_series = change_finder.show()
        assert isinstance(phase_series, PhaseSeries)
