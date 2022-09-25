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
        snr_build = ODEScenario.from_json(filename=jsonpath)
        assert snr == snr_build
        with pytest.raises(NotImplemented):
            snr == "a"
        with pytest.raises(AssertionError):
            snr_build._location_name = "unknown"
            assert snr == snr_build
