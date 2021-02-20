#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import Term, PopulationData


class TestPopulationData(object):
    def test_cleaning(self, population_data):
        df = population_data.cleaned()
        column_set = set(Term.STR_COLUMNS) | set([Term.N, Term.ISO3])
        assert set(df.columns) == column_set

    def test_total(self, population_data):
        assert isinstance(population_data.total(), int)

    def test_to_dict(self, population_data):
        assert isinstance(population_data.to_dict(country_level=True), dict)
        assert isinstance(population_data.to_dict(country_level=False), dict)

    def test_value(self, population_data):
        assert isinstance(population_data.value("Japan"), int)
        old_value = population_data.value("Japan", date="01Mar2020")
        assert isinstance(old_value, int)
        with pytest.raises(KeyError):
            population_data.value("Japan", "01Jan1000")
        population_data.value("UK")

    def test_update(self):
        population_data = PopulationData(filename=None)
        population_data.update(1000, "Moon")
        assert population_data.value("Moon") == 1000
        population_data.update(2000, "Moon")
        assert population_data.value("Moon") == 2000

    def test_countries(self, population_data):
        assert isinstance(population_data.countries(), list)

    def test_map(self, population_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        population_data.map(country=None)
        population_data.map(country="Japan")
        with pytest.raises(NotImplementedError):
            population_data.map(variable="Feeling")
