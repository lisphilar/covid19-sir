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
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        change_finder = ChangeFinder(sr_df)
        change_finder.run()
        change_finder.show()
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
            country="Italy", population=population, end_date="23Jan2020")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
        trend.show()

    def test_one_phase_with_few_records(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(
            country="Italy", population=population,
            start_date="25Mar2020", end_date="26Mar2020"
        )
        with pytest.raises(ValueError):
            trend = Trend(sr_df)
            trend.run()

    def test_one_phase_rmsle_without_analyse(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(
            country="Italy", population=population,
            start_date="25Mar2020", end_date="02Apr2020"
        )
        trend = Trend(sr_df)
        with pytest.raises(NameError):
            trend.rmsle()

    def test_one_phase_show_without_analyse(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(
            country="Italy", population=population,
            start_date="25Mar2020", end_date="02Apr2020"
        )
        trend = Trend(sr_df)
        with pytest.raises(NameError):
            trend.show()
