#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import ChangeFinder
from covsirphy import PhaseSeries


class TestChangeFinder(object):
    def test_find(self, jhu_data, population_data):
        clean_df = jhu_data.cleaned()
        population = population_data.value("Italy")
        change_finder = ChangeFinder(
            clean_df, population, country="Italy"
        )
        change_finder.run()
        phase_series = change_finder.show()
        assert isinstance(phase_series, PhaseSeries)
