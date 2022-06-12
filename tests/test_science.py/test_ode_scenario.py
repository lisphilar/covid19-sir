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
        snr.simulate()
        snr.simulate(name="Baseline")
