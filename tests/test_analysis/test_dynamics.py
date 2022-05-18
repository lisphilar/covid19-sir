#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import Dynamics, SIR, SIRD, SIRF, UnExecutedError


class TestDynamics(object):
    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF])
    def test_generate(self, model, imgfile):
        dynamics = Dynamics.from_sample(model=model, first_date="01Jan2020", last_date="31Dec2020")
        dynamics.segment(points=["01May2020", "01Sep2020"])
        with pytest.raises(UnExecutedError):
            dynamics.estimate()
        rho_eg = model.EXAMPLE["param_dict"]["rho"]
        rho0 = dynamics.get(date="01Jan2020", variable="rho")
        # 1st phase
        dynamics.update(start_date="01May2020", end_date="30Sep2020", variable="rho", value=rho0 * 2)
        assert dynamics.get(date="01May2020", variable="rho") == rho_eg * 2
        # 2nd phase
        dynamics.update(start_date="01May2020", end_date="30Sep2020", variable="rho", value=rho0 * 3)
        dynamics.update(start_date="01May2020", end_date="30Sep2020", variable="Infected", value=1000)
        assert dynamics.get(date="01May2020", variable="rho") == rho_eg * 3
        # Check simulation, tracking, summary
        assert set(model.VARIABLES).issubset(dynamics.simulate(ffill=True, model_specific=True))
        assert dynamics.track(ffill=False).isna().any().any()
        summary_df = dynamics.summary(ffill=True)
        assert not summary_df.isna().any().any()
        assert len(summary_df) == 3
        # S-R trend analysis after simulation
        dynamics.sr(simulated=True, filename=imgfile)
