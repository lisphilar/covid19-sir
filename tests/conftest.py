#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataLoader


@pytest.fixture(scope="session")
def data_loader():
    return DataLoader()


@pytest.fixture(scope="session")
def jhu_data(data_loader):
    return data_loader.jhu()


@pytest.fixture(scope="session")
def population_data(data_loader):
    return data_loader.population()


@pytest.fixture(scope="session")
def oxcgrt_data(data_loader):
    return data_loader.oxcgrt()


@pytest.fixture(scope="session")
def japan_data(data_loader):
    return data_loader.japan()


@pytest.fixture(scope="session")
def linelist_data(data_loader):
    return data_loader.linelist()


@pytest.fixture(scope="session")
def pcr_data(data_loader):
    return data_loader.pcr()
