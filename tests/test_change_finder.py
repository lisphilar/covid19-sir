#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import ChangeFinder, Trend
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


class TestTrend(object):
    def test_one_phase(self, jhu_data, population_data):
        population = population_data.value("Italy")
        trend = Trend(
            jhu_data, population, "Italy",
            start_date="25Mar2020", end_date="02Apr2020"
        )
        trend.analyse()
        assert isinstance(trend.rmsle(), float)
        trend.show()

    def test_one_phase_with_dataframe(self, jhu_data, population_data):
        population = population_data.value("Italy")
        clean_df = jhu_data.cleaned()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        trend = Trend(
            clean_df, population, "Italy",
            start_date="25Mar2020", end_date="02Apr2020"
        )
        trend.analyse()
        assert isinstance(trend.rmsle(), float)
        trend.show()

    def test_one_phase_with_few_records(self, jhu_data, population_data):
        population = population_data.value("Italy")
        clean_df = jhu_data.subset("Italy")
        clean_df[Term.COUNTRY] = "Italy"
        clean_df[Term.PROVINCE] = Term.UNKNOWN
        clean_df = clean_df.iloc[:2, :]
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with pytest.raises(ValueError):
            _ = Trend(clean_df, population, "Italy")

    def test_one_phase_rmsle_without_analyse(self, jhu_data, population_data):
        population = population_data.value("Italy")
        trend = Trend(
            jhu_data, population, "Italy",
            start_date="25Mar2020", end_date="02Apr2020"
        )
        with pytest.raises(NameError):
            trend.rmsle()

    def test_one_phase_show_without_analyse(self, jhu_data, population_data):
        population = population_data.value("Italy")
        trend = Trend(
            jhu_data, population, "Italy",
            start_date="25Mar2020", end_date="02Apr2020"
        )
        with pytest.raises(NameError):
            trend.show()
