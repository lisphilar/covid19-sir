#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import ODEScenario, SIRFModel, SubsetNotFoundError


class TestODEScenario(object):
    def test_auto_build(self):
        snr = ODEScenario.auto_build(geo="Japan", model=SIRFModel)
        with pytest.raises(SubsetNotFoundError):
            ODEScenario.auto_build(geo="Moon", model=SIRFModel)
        snr.summary()
        snr.track()
        snr.simulate()
        snr.simulate(name="Baseline")
        snr.build_with_template(name="Lockdown", template="Baseline")
        snr.append(end=30, name="Lockdown", rho=0.1)
        snr.build_with_template(name="Lockdown2", template="Baseline")
        snr.delete(name="Lockdown2")
        snr.summary()
