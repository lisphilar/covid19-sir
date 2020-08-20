#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import ChangeFinder, Trend


class TestChangeFinder(object):
    @pytest.mark.parametrize(
        "country",
        ["Italy", "Japan", "United States", "India", "New Zealand"]
    )
    def test_find(self, jhu_data, population_data, country):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        change_finder = ChangeFinder(sr_df)
        change_finder.run()
        warnings.filterwarnings("ignore", category=UserWarning)
        change_finder.show(area=jhu_data.area_name(country="Italy"))
        assert isinstance(change_finder.date_range(), tuple)

    def test_find_with_small_min_size(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(country="Italy", population=population)
        with pytest.raises(ValueError):
            change_finder = ChangeFinder(sr_df, min_size=2)
            change_finder.run()

    def test_find_with_few_records(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(
            country="Italy", population=population, end_date="24Feb2020")
        with pytest.raises(ValueError):
            min_size = 7
            change_finder = ChangeFinder(sr_df, min_size=min_size)
            change_finder.run()


class TestTrend(object):
    def test_one_phase(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(
            country="Italy", population=population,
            start_date="25Mar2020", end_date="02Apr2020"
        )
        trend = Trend(sr_df)
        trend.run()
        assert isinstance(trend.rmsle(), float)
        warnings.filterwarnings("ignore", category=UserWarning)
        trend.show(area=jhu_data.area_name(country="Italy"))

    def test_one_phase_with_few_records(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(
            country="Italy", population=population,
            start_date="25Mar2020", end_date="26Mar2020"
        )
        with pytest.raises(ValueError):
            trend = Trend(sr_df)
            trend.run()
