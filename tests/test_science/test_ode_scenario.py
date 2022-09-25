#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import pytest
from covsirphy import ODEScenario, SIRFModel, ScenarioNotFoundError, Term


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
    def test_scenario_manipulation(self, snr):
        snr.build_with_template(name="New1", template="Baseline")
        with pytest.raises(ScenarioNotFoundError):
            snr.build_with_template(name="New2", template="Unknown")
        with pytest.raises(ScenarioNotFoundError):
            snr.to_dynamics(name="New2")
        snr.build_with_template(name="New2", template="Baseline")
        snr.build_with_template(name="Old", template="Baseline")
        snr.build_with_template(name="Wow", template="Baseline")
        snr.delete(pattern="Old", exact=True)
        snr.delete(pattern="New", exact=False)
        snr.rename(old="Wow", new="Excellent")
        assert set(snr.track()[Term.SERIES].unique()) == {"Baseline", "Excellent"}
        assert isinstance(snr.summary(), pd.DataFrame)
        snr.delete(pattern="Excellent", exact=True)

    def test_auto_filed(self, snr):
        with pytest.raises(ScenarioNotFoundError):
            ODEScenario.auto_build(geo="Moon", model=SIRFModel)
