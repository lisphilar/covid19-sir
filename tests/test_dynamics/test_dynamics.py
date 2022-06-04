#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
from covsirphy import Dynamics, SIRFModel, Term, Validator
from covsirphy import EmptyError, NotEnoughDataError, NAFoundError, UnExpectedNoneError


@pytest.mark.parametrize("model", [SIRFModel])
class TestDynamics(object):
    def test_one_phase(self, model):
        dyn = Dynamics.from_sample(model)
        del dyn.name
        dyn.name = "Dynamics"
        assert dyn.name == "Dynamics"
        registered_df = dyn.register()
        Validator(registered_df).dataframe(columns=[*SIRFModel._VARIABLES, *SIRFModel._PARAMETERS])
        summary_df = dyn.summary()
        assert len(summary_df) == 1
        Validator(dyn.simulate()).dataframe(columns=[Term.S, Term.CI, Term.F, Term.R])
        Validator(dyn.simulate(model_specific=True)).dataframe(columns=SIRFModel._VARIABLES)
        assert dyn.model_name == model._NAME

    def test_two_phase(self, model):
        dyn = Dynamics.from_sample(model)
        registered_df = dyn.register()
        with pytest.raises(EmptyError):
            failed_df = registered_df.copy()
            failed_df[Term.CI] = np.nan
            dyn.register(failed_df)
        registered_df.loc[registered_df.index[90], "rho"] = registered_df.loc[registered_df.index[0], "rho"] * 2
        dyn.register(data=registered_df)
        summary_df = dyn.summary()
        assert len(summary_df) == 2
        Validator(dyn.simulate()).dataframe(columns=[Term.S, Term.CI, Term.F, Term.R])

    def test_from_data(self, model):
        sample_dyn = Dynamics.from_sample(model)
        dyn = Dynamics.from_data(model=model, data=sample_dyn.simulate())
        df = dyn.register()
        assert df.loc[df.index[0], model._PARAMETERS[0]] is pd.NA

    def test_simulate_failed(self, model):
        dyn = Dynamics.from_sample(model)
        df = dyn.register()
        df[model._PARAMETERS[0]] = np.nan
        dyn.register(data=df)
        with pytest.raises(NAFoundError):
            dyn.simulate()

    def test_trend(self, model, imgfile):
        with pytest.raises(NotEnoughDataError):
            dyn_failed = Dynamics.from_sample(model=model)
            dyn_failed.trend_analysis()
        dyn = Dynamics.from_sample(model)
        dyn.register(dyn.simulate())
        points, df = dyn.trend_analysis(filename=imgfile)
        assert isinstance(points, list)
        assert {"Actual", "0th"}.issubset(df.columns)
        dyn.segment(filename=imgfile)
        assert len(dyn) > 1

    def test_estimate(self, model):
        dyn = Dynamics.from_sample(model)
        dyn.register(dyn.simulate())
        dyn.segment()
        dyn.tau = 1440
        del dyn.tau
        assert dyn.tau is None
        with pytest.raises(UnExpectedNoneError):
            dyn.simulate()
        with pytest.raises(UnExpectedNoneError):
            dyn.estimate_params()
        dyn.estimate().summary()
        assert dyn.tau is not None
        dyn.simulate()
