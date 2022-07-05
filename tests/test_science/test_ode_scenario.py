#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import ODEScenario, SIRFModel, SubsetNotFoundError, ScenarioNotFoundError, Term


@pytest.fixture(scope="module")
def snr():
    return ODEScenario.auto_build(geo="Japan", model=SIRFModel)


class TestODEScenario(object):
    def test_with_template_failed(self, snr):
        with pytest.raises(ScenarioNotFoundError):
            snr.build_with_template(name="New", template="Un-registered")

    def test_auto_build_failed(self):
        with pytest.raises(SubsetNotFoundError):
            ODEScenario.auto_build(geo="Moon", model=SIRFModel)

    def test_to_dynamics_failed(self, snr):
        with pytest.raises(ScenarioNotFoundError):
            snr.to_dynamics(name="Un-registered")

    def test_simulate(self, snr, imgfile):
        snr.simulate(filename=imgfile)
        snr.simulate(name="Baseline", display=False)

    def test_build_delete(self, snr):
        snr.build_with_template(name="Lockdown", template="Baseline")
        with pytest.raises(ScenarioNotFoundError):
            snr.append(name="Un-registered")
        snr.append(end=30, name="Lockdown", rho=0.1)
        snr.build_with_template(name="Lockdown2", template="Baseline")
        snr.build_with_template(name="Lockdown3", template="Baseline")
        assert {"Lockdown", "Lockdown2", "Lockdown3"}.issubset(snr.summary().reset_index()[snr.SERIES].unique())
        snr.delete(pattern="Lockdown3")
        assert not {"Lockdown3"}.issubset(snr.summary().reset_index()[snr.SERIES].unique())
        snr.delete(pattern="Lockdown^")
        assert not {"Lockdown", "Lockdown2", "Lockdown3"}.issubset(snr.summary().reset_index()[snr.SERIES].unique())
        snr.build_with_template(name="Medicine", template="Baseline")
        snr.append(end="01Jan2100", name="Medicine", sigma=0.5)
        snr.append()
        df = snr.summary().reset_index().groupby(Term.SERIES).last()
        assert len(df[Term.END].unique()) == 1

    def test_compare(self, snr):
        snr.build_with_template(name="Lockdown", template="Baseline")
        snr.append(end=30, name="Medicine", sigma=0.5)
        snr.append()
        snr.compare_cases(variable=Term.C)
        snr.compare_param(param="rho")
        snr.compare_param(param="Rt")
        snr.compare_param(param="1/gamma [day]")

    def test_summary_track_describe(self, snr):
        snr.summary()
        snr.track()
        snr.describe()

    def test_predict(self, snr):
        snr.predict(days=30, name="Baseline")
        snr.rename(old="Baseline_Univariate_Likely", new="Likely")
        snr.represent(q=(0.1, 0.9), excluded=["Baseline"])
