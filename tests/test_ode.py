#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import Term, UnExecutedError, find_args
from covsirphy import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF, ODEHandler


class TestODEHandler(object):
    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF, SEWIRF])
    @pytest.mark.parametrize("first_date", ["01Jan2021"])
    @pytest.mark.parametrize("tau", [720])
    def test_simulate(self, model, first_date, tau):
        y0_dict = model.EXAMPLE["y0_dict"]
        param_dict = model.EXAMPLE["param_dict"]
        handler = ODEHandler(model, first_date, tau)
        handler.add(end_date="31Jan2021", y0_dict=y0_dict, param_dict=param_dict)
        handler.add(end_date="28Feb2021", y0_dict=None, param_dict=param_dict)
        sim_df = handler.simulate().set_index(Term.DATE)
        assert sim_df.index.min() == pd.to_datetime(first_date)
        assert sim_df.index.max() == pd.to_datetime("28Feb2021")
        assert set(sim_df.reset_index().columns) == set(Term.DSIFR_COLUMNS)

    @pytest.mark.parametrize("model", [SIR])
    @pytest.mark.parametrize("first_date", ["01Jan2021"])
    @pytest.mark.parametrize("tau", [720])
    def test_simulate_error(self, model, first_date, tau):
        y0_dict = model.EXAMPLE["y0_dict"]
        handler = ODEHandler(model, first_date, tau)
        with pytest.raises(UnExecutedError):
            handler.simulate()
        with pytest.raises(ValueError):
            handler.add(end_date="31Jan2021", y0_dict=None)
        handler.add(end_date="31Jan2021", y0_dict=y0_dict, param_dict=None)
        with pytest.raises(ValueError):
            handler.simulate()

    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF])
    @pytest.mark.parametrize("first_date", ["01Jan2021"])
    @pytest.mark.parametrize("tau", [720])
    @pytest.mark.parametrize("n_jobs", [1, -1])
    def test_estimate(self, model, first_date, tau, n_jobs):
        # Create simulated dataset
        y0_dict = model.EXAMPLE["y0_dict"]
        param_dict = model.EXAMPLE["param_dict"]
        sim_handler = ODEHandler(model, first_date, tau)
        sim_handler.add(end_date="31Jan2021", y0_dict=y0_dict, param_dict=param_dict)
        sim_handler.add(end_date="28Feb2021", y0_dict=None, param_dict=param_dict)
        sim_df = sim_handler.simulate()
        # Set-up handler
        handler = ODEHandler(model, first_date, tau=None, metric="RMSLE", n_jobs=n_jobs)
        with pytest.raises(UnExecutedError):
            handler.estimate_tau(sim_df)
        with pytest.raises(UnExecutedError):
            handler.estimate_params(sim_df)
        handler.add(end_date="31Jan2021", y0_dict=y0_dict)
        handler.add(end_date="28Feb2021")
        # Simulation needs tau setting
        with pytest.raises(UnExecutedError):
            handler.simulate()
        # Estimate tau and ODE parameters
        with pytest.raises(UnExecutedError):
            handler.estimate_params(sim_df)
        tau_est, info_dict_est = handler.estimate(sim_df, timeout=1, timeout_iteration=1)
        assert isinstance(tau_est, int)
        assert isinstance(info_dict_est, dict)

    @pytest.mark.parametrize("model", [SIR])
    def test_model_common(self, model):
        model_ins = model(population=1_000_000, rho=0.2, sigma=0.075)
        assert str(model_ins) == "SIR model with rho=0.2, sigma=0.075"
        assert model_ins["rho"] == 0.2
        with pytest.raises(KeyError):
            assert model_ins["kappa"] == 0.1

    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF, SEWIRF])
    def test_irregular_params(self, model):
        param_dict = model.EXAMPLE[Term.PARAM_DICT].copy()
        param_dict["sigma"] = 0
        param_dict["kappa"] = 0
        model_ins = model(population=1_000_000, **find_args(model, **param_dict))
        assert str(model_ins)
        assert model_ins.calc_r0() is None
        assert model_ins.calc_days_dict(tau=1440)

    @pytest.mark.parametrize("model", [ModelBase])
    def test_model_base(self, model):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model_ins = model(population=1_000_000)
        with pytest.raises(NotImplementedError):
            model_ins(1, [0, 0, 0])
        with pytest.raises(NotImplementedError):
            model.param_range(1, 2)
        with pytest.raises(NotImplementedError):
            model.specialize(1, 2)
        with pytest.raises(NotImplementedError):
            model.restore(1)
        with pytest.raises(NotImplementedError):
            model.convert(1, 2)
        with pytest.raises(NotImplementedError):
            model.convert_reverse(1, 2, 3)
        with pytest.raises(NotImplementedError):
            model.guess(1, 2)
        with pytest.raises(NotImplementedError):
            model_ins.calc_r0()
        with pytest.raises(NotImplementedError):
            model_ins.calc_days_dict(1440)

    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF])
    def test_deprecated_class_methods(self, model, jhu_data, population_data):
        # Set-up
        population = population_data.value("Japan")
        subset_df = jhu_data.subset("Japan", population=population)
        taufree_df = model.convert(subset_df, tau=1440).reset_index()
        # Test
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model.tau_free(subset_df, population, tau=None)
        model.tau_free(subset_df, population, tau=1440)
        specialized_df = model.specialize(subset_df, population)
        model.restore(specialized_df)
        model.param_range(taufree_df, population)

    @pytest.mark.parametrize("model", [SIRFV])
    def test_deprecated(self, model):
        with pytest.raises(NotImplementedError):
            model(1, 1, 1, 1, 1)

    @pytest.mark.parametrize("model", [SEWIRF])
    def test_estimation_deprecated(self, model, jhu_data, population_data):
        # Set-up
        population = population_data.value("Japan")
        subset_df = jhu_data.subset("Japan", population=population)
        # Test
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        specialized_df = model.specialize(subset_df, population)
        model.restore(specialized_df)
        model.convert(subset_df, tau=1440)
        model.convert(subset_df, tau=None)
        with pytest.raises(NotImplementedError):
            model.guess(1, 2)
        with pytest.raises(NotImplementedError):
            model.param_range(1, 2)
        param_dict = model.EXAMPLE[Term.PARAM_DICT].copy()
        model_ins = model(population=1_000_000, **param_dict)
        assert model_ins.calc_r0()
        assert model_ins.calc_days_dict(tau=1440)
