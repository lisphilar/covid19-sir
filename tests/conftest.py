#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataLoader, PhaseSeries, ParamTracker


@pytest.fixture(scope="session")
def jhu_data():
    data_loader = DataLoader()
    return data_loader.jhu()


@pytest.fixture(scope="session")
def data_loader():
    return DataLoader()


@pytest.fixture(scope="session")
def population_data():
    data_loader = DataLoader()
    return data_loader.population()


@pytest.fixture(scope="session")
def oxcgrt_data():
    data_loader = DataLoader()
    return data_loader.oxcgrt()


@pytest.fixture(scope="session")
def japan_data():
    data_loader = DataLoader()
    return data_loader.japan()


@pytest.fixture(scope="session")
def param_tracker():
    data_loader = DataLoader()
    jhu_data = data_loader.jhu()
    population_data = data_loader.population()
    population = population_data.value(country="Japan")
    record_df = jhu_data.subset(country="Japan", population=population)
    series = PhaseSeries("01Apr2020", "01Nov2020", population)
    return ParamTracker(
        record_df=record_df, phase_series=series, area="Japan"
    )
