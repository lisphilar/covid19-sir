#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import ExampleData, SIRF


class TestExampleData(object):
    def test_iso3(self):
        example_data = ExampleData()
        example_data.add(SIRF, country="Japan")
        assert example_data.country_to_iso3("Japan") == "JPN"
        example_data.add(SIRF, country="Moon")
        assert example_data.country_to_iso3("Moon") == "---"

    def test_subset(self):
        example_data = ExampleData()
        example_data.add(SIRF, country="Japan")
        example_data.subset(country="Japan")
        example_data.subset_complement(country="Japan")
        example_data.records(country="Japan")
