#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import ExampleData, SIRF


class TestExampleData(object):
    def test_iso3(self):
        example_data = ExampleData()
        example_data.add(SIRF, country="Japan")
        assert example_data.country_to_iso3("Japan") == "JPN"
        example_data.add(SIRF, country="Moon")
        assert example_data.country_to_iso3("Moon") == "---"

    def test_one_phase(self):
        example_data = ExampleData()
        example_data.add(SIRF)
        with pytest.raises(ValueError):
            example_data.subset()
        example_data.subset(model=SIRF)
        example_data.subset(country=SIRF.NAME)
        example_data.subset_complement(model=SIRF)
        example_data.records(model=SIRF)
        example_data.specialized(model=SIRF)
        example_data.non_dim(model=SIRF)

    def test_two_phases(self):
        example_data = ExampleData()
        example_data.add(SIRF)
        example_data.add(SIRF)
        example_data.subset(model=SIRF)
