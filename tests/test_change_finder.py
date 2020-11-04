#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import ChangeFinder, Trend, Term


class TestChangeFinder(object):
    @pytest.mark.parametrize(
        "country",
        [
            "Italy", "Japan", "India", "United States", "Greece", "Russia",
            "Brazil", "France", "Spain", "UK", "New Zealand", "Germany",
        ]
    )
    def test_find(self, jhu_data, population_data, country, max_rmsle=20.0):
        # Setup
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        # Find change points
        change_finder = ChangeFinder(sr_df, max_rmsle=max_rmsle)
        change_finder.run()
        # For all phases, check if RMSLE score is lower than max_rmsle=20.0
        for (start_date, end_date) in zip(*change_finder.date_range()):
            phase_df = jhu_data.to_sr(
                country=country, population=population,
                start_date=start_date, end_date=end_date)
            rmsle = Trend(sr_df=phase_df).rmsle()
            assert rmsle < max_rmsle

    @pytest.mark.parametrize("country", ["Italy"])
    def test_show(self, jhu_data, population_data, country):
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
    @pytest.mark.parametrize("country", ["Italy"])
    @pytest.mark.parametrize("func", ["linear", "negative exponential"])
    def test_one_phase(self, jhu_data, population_data, country, func):
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(
            country=country, population=population,
            start_date="25Mar2020", end_date="02Apr2020")
        trend = Trend(sr_df)
        assert isinstance(trend.rmsle(), float)
        trend.show(area=jhu_data.area_name(country="Italy"))

    @pytest.mark.parametrize("country", ["Italy"])
    def test_one_phase_with_few_records(self, jhu_data, population_data, country):
        warnings.simplefilter("ignore", category=UserWarning)
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(
            country=country, population=population,
            start_date="25Mar2020", end_date="26Mar2020")
        with pytest.raises(ValueError):
            Trend(sr_df).run()
