#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import TrendDetector


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

    @pytest.mark.parametrize(
        "country",
        [
            "Italy", "Japan", "India", "United States", "Greece", "Russia",
            "Brazil", "France", "Spain", "UK", "New Zealand", "Germany",
        ]
    )
    def test_sr(self, jhu_data, population_data, country, imgfile):
        # Dataset
        population = population_data.value(country)
        subset_df = jhu_data.subset(country=country, population=population)
        # S-R trend analysis
        detector = TrendDetector(data=subset_df, area=country)
        detector.sr()
        # Summary
        detector.summary()
        # Show plane
        detector.sr_show(filename=imgfile)
