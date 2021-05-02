#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.term import Term
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
        # Subset
        subset_df = example_data.subset(model=SIRF)
        assert subset_df.columns.tolist() == Term.SUB_COLUMNS
        example_data.subset(country=SIRF.NAME)
        example_data.subset_complement(model=SIRF)
        example_data.records(model=SIRF)
        # Specialized
        specialized_df = example_data.specialized(model=SIRF)
        assert specialized_df.columns.tolist() == [Term.DATE, *SIRF.VARIABLES]
        # Non-dimensional
        nondim_df = example_data.non_dim(model=SIRF)
        assert nondim_df.columns.tolist() == SIRF.VARIABLES
        assert round(nondim_df.sum().sum()) == len(nondim_df)

    def test_two_phases(self):
        example_data = ExampleData()
        example_data.add(SIRF)
        example_data.add(SIRF)
        example_data.subset(model=SIRF)
