#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataLoader


@pytest.fixture(autouse=True)
def jhu_data():
    data_loader = DataLoader()
    return data_loader.jhu()


@pytest.fixture(autouse=True)
def population_data():
    data_loader = DataLoader()
    return data_loader.population()


@pytest.fixture(autouse=True)
def oxcgrt_data():
    data_loader = DataLoader()
    return data_loader.oxcgrt()


@pytest.fixture(autouse=True)
def japan_data():
    data_loader = DataLoader()
    return data_loader.japan()
