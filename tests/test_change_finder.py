#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import ChangeFinder
from covsirphy import PhaseSeries, Term


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

    def test_find_with_small_min_size(self, jhu_data, population_data):
        min_size = 2
        population = population_data.value("Italy")
        clean_df = jhu_data.cleaned()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with pytest.raises(ValueError):
            change_finder = ChangeFinder(
                clean_df, population, country="Italy",
                min_size=min_size
            )
            change_finder.run()

    def test_find_with_few_records(self, jhu_data, population_data):
        min_size = 7
        population = population_data.value("Italy")
        clean_df = jhu_data.subset("Italy")
        clean_df[Term.COUNTRY] = "Italy"
        clean_df[Term.PROVINCE] = Term.UNKNOWN
        clean_df = clean_df.iloc[:(min_size - 1), :]
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with pytest.raises(ValueError):
            change_finder = ChangeFinder(
                clean_df, population, country="Italy",
                min_size=min_size
            )
            change_finder.run()
