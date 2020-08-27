#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import ChangeFinder, Trend, Term


class TestChangeFinder(object):
    @pytest.mark.parametrize(
        "country",
        ["Italy", "Japan", "United States", "India", "New Zealand"]
    )
    def test_find(self, jhu_data, population_data, country):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        change_finder = ChangeFinder(sr_df)
        change_finder.run()
        change_finder.show(area=jhu_data.area_name(country=country))
        assert isinstance(change_finder.date_range(), tuple)

    def test_find_with_small_min_size(self, jhu_data, population_data):
        population = population_data.value("Italy")
        sr_df = jhu_data.to_sr(country="Italy", population=population)
        with pytest.raises(ValueError):
            change_finder = ChangeFinder(sr_df, min_size=2)
            change_finder.run()

    def test_find_with_few_records(self, jhu_data, population_data):
        population = population_data.value("Italy")
        min_size = 7
        df = jhu_data.subset(country="Italy")
        start_date = df.loc[df.index[0], Term.DATE].strftime(Term.DATE_FORMAT)
        end_date = Term.date_change(start_date, days=min_size - 2)
        sr_df = jhu_data.to_sr(
            country="Italy", population=population, end_date=end_date)
        with pytest.raises(ValueError):
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
        warnings.simplefilter("ignore", category=UserWarning)
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
