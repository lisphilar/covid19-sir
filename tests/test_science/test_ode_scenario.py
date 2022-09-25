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
    return ODEScenario.auto_build(geo="Japan", model=SIRFModel)


class TestODEScenario(object):
    def test_json(self, jsonpath, snr):
        snr.to_json(filename=jsonpath)
        assert snr == ODEScenario.from_json(filename=jsonpath)
