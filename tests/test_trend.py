#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import TrendDetector, Trend, ChangeFinder, UnExpectedValueError


class TestTrendDetector(object):

    @pytest.mark.parametrize("country", ["Japan"])
    def test_min_size(self, jhu_data, population_data, country):
        # Dataset
        population = population_data.value(country)
        subset_df = jhu_data.subset(country=country, population=population)
        # Too small min_size
        with pytest.raises(ValueError):
            TrendDetector(data=subset_df, min_size=2)
        # Too large min_size
        with pytest.raises(ValueError):
            TrendDetector(data=subset_df, min_size=100000)
        # Reset
        detector = TrendDetector(data=subset_df)
        detector.reset()
        # Deprecated
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ChangeFinder(subset_df)
        Trend(subset_df)

    @pytest.mark.parametrize(
        "algo", ["Pelt-rbf", "Binseg-rbf", "Binseg-normal", "BottomUp-rbf", "BottomUp-normal"])
    @pytest.mark.parametrize(
        "country",
        [
            "Italy", "India", "USA", "Greece", "Russia",
            "Brazil", "France", "Spain", "UK", "New Zealand", "Germany",
        ]
    )
    def test_sr(self, jhu_data, population_data, algo, country):
        # Dataset
        population = population_data.value(country)
        record_df, _ = jhu_data.records(country=country, population=population)
        # S-R trend analysis
        detector = TrendDetector(data=record_df, area=country)
        detector.sr(algo=algo)
        # Summary
        detector.summary()

    @pytest.mark.parametrize("algo", ["Unknown"])
    @pytest.mark.parametrize("country", ["Japan"])
    def test_sr_algo_error(self, jhu_data, population_data, algo, country):
        # Dataset
        population = population_data.value(country)
        subset_df = jhu_data.subset(country=country, population=population)
        # S-R trend analysis
        detector = TrendDetector(data=subset_df, area=country)
        with pytest.raises(UnExpectedValueError):
            detector.sr(algo=algo)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_show(self, jhu_data, population_data, country, imgfile):
        # Dataset
        population = population_data.value(country)
        subset_df = jhu_data.subset(country=country, population=population)
        # Without change points
        detector = TrendDetector(data=subset_df, area=country)
        detector.show(filename=imgfile, show_legend=False)
        # S-R trend analysis
        detector.sr()
        # Summary
        detector.summary()
        # Show plane
        detector.show(filename=imgfile)
