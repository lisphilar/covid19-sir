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
        clean_df = jhu_data.cleaned()
        population = population_data.value(country)
        change_finder = ChangeFinder(
            clean_df, population, country=country
        )
        change_finder.run()
        phase_series = change_finder.show()
        assert isinstance(phase_series, PhaseSeries)
