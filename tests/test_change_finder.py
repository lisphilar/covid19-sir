#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import ChangeFinder
from covsirphy import PhaseSeries


class TestChangeFinder(object):
    @pytest.mark.parametrize(
        "country",
        ["Italy", "Japan", "United States", "India", "New Zealand"]
    )
    def test_find(self, jhu_data, population_data, country):
        population = population_data.value(country)
        clean_df = jhu_data.subset(country, population=population)
        change_finder = ChangeFinder(
            clean_df, population, country=country
        )
        change_finder.run()
        phase_series = change_finder.show()
        assert isinstance(phase_series, PhaseSeries)
