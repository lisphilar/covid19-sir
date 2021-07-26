#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    return data_loader.population()


@pytest.fixture(scope="session")
def oxcgrt_data(data_loader):
    return data_loader.oxcgrt()


@pytest.fixture(scope="session")
def japan_data(data_loader):
    return data_loader.japan()


@pytest.fixture(scope="session")
def pcr_data(data_loader):
    return data_loader.pcr()


@pytest.fixture(scope="session")
def vaccine_data(data_loader):
    return data_loader.vaccine()


@pytest.fixture(scope="session")
def mobility_data(data_loader):
    return data_loader.mobility()


@pytest.fixture(scope="session")
def pyramid_data(data_loader):
    return data_loader.pyramid()


@pytest.fixture(
    scope="session",
    params=[
        "jhu_data", "population_data", "oxcgrt_data", "japan_data",
        "pcr_data", "vaccine_data", "mobility_data", "pyramid_data",
    ])
def data(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def imgfile():
    dirpath = Path("input")
    dirpath.mkdir(exist_ok=True)
    filepath = dirpath.joinpath("test.jpg")
    yield str(filepath)
    try:
        filepath.unlink(missing_ok=True)
    except TypeError:
        # Python 3.7
        if filepath.exists():
            filepath.unlink()
