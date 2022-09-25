#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
from covsirphy import ODEScenario, SIRFModel


@pytest.fixture(scope="module")
def jsonpath():
    dirpath = Path("input")
    dirpath.mkdir(exist_ok=True)
    jsonpath = dirpath.joinpath("test.json")
    yield jsonpath
    jsonpath.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def snr(jsonpath):
    if jsonpath.exists():
        return ODEScenario.from_json(filename=jsonpath)
    instance = ODEScenario.auto_build(geo="Japan", model=SIRFModel)
    instance.to_json(filename=jsonpath)
    snr_build = ODEScenario.from_json(filename=jsonpath)
    assert instance._location_name == snr_build._location_name
    assert instance.describe().equals(snr_build.describe())
    assert instance == snr_build
    snr_build._location_name = "unknown"
    assert snr != snr_build
    return instance


class TestODEScenario(object):
    def test_scenario_manipulation(self):
        pass
