#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import ExampleData, PopulationData, Term, ModelValidator, UnExecutedError
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
        tau_est, info_dict_est = handler.estimate(sim_df, timeout=5)
        assert isinstance(tau_est, int)
        assert isinstance(info_dict_est, dict)


class TestODE(object):
    @pytest.mark.parametrize(
        "model",
        [SIR, SIRD, SIRF, SEWIRF])
    def test_ode(self, model):
        # Setting
        eg_tau = 1440
        area = {"country": "Full", "province": model.NAME}
        # Population
        population_data = PopulationData(filename=None)
        population_data.update(model.EXAMPLE["population"], **area)
        # Simulation
        example_data = ExampleData(tau=eg_tau, start_date="01Jan2020")
        example_data.add(model, **area)
        # Model-specialized records
        spe_df = example_data.specialized(**area)
        assert set(spe_df.columns) == set(
            [*Term.STR_COLUMNS, *model.VARIABLES])
        # Non-dimensional records
        nondim_df = example_data.non_dim(**area)
        assert set(nondim_df.columns) == set(
            [Term.TS, *list(model.VAR_DICT.keys())])
        # JHU-type records
        jhu_df = example_data.subset(**area)
        assert set(jhu_df.columns) == set(Term.NLOC_COLUMNS)
        # Calculate Rt/day parameters when parameters are None
        param_dict = {p: 0 for p in model.PARAMETERS}
        model_instance = model(population_data.value(**area), **param_dict)
        model_instance.calc_r0()
        model_instance.calc_days_dict(eg_tau)

    @pytest.mark.parametrize("model", [SIR])
    def test_model_common(self, model):
        model_ins = model(population=1_000_000, rho=0.2, sigma=0.075)
        assert str(model_ins) == "SIR model with rho=0.2, sigma=0.075"
        assert model_ins["rho"] == 0.2
        with pytest.raises(KeyError):
            assert model_ins["kappa"] == 0.1

    @pytest.mark.parametrize("model", [ModelBase])
    def test_model_base(self, model):
        model_ins = model(population=1_000_000)
        with pytest.raises(NotImplementedError):
            model_ins(1, [0, 0, 0])
        with pytest.raises(NotImplementedError):
            model.param_range(1, 2)
        with pytest.raises(NotImplementedError):
            model.specialize(1, 2)
        with pytest.raises(NotImplementedError):
            model_ins.calc_r0()
        with pytest.raises(NotImplementedError):
            model_ins.calc_days_dict(1440)

    @pytest.mark.parametrize("model", [SIR])
    def test_error(self, model):
        # Setting
        eg_tau = 1440
        # Simulation
        example_data = ExampleData(tau=eg_tau, start_date="01Jan2020")
        with pytest.raises(KeyError):
            assert not example_data.specialized(model=model).empty
        with pytest.raises(KeyError):
            assert not example_data.non_dim(model=model).empty
        example_data.add(model)
        # Model-specialized records
        with pytest.raises(ValueError):
            assert not example_data.specialized().empty

    def test_ode_two_phases(self, population_data):
        # Setting
        eg_tau = 1440
        area = {"country": "Example", "province": "Example"}
        # Simulation
        example_data = ExampleData(tau=eg_tau)
        example_data.add(SIRF, step_n=30, **area)
        example_data.add(SIRD, step_n=30, **area)
        dim_df = example_data.subset(**area)
        assert isinstance(dim_df, pd.DataFrame)
        assert set(dim_df.columns) == set(Term.NLOC_COLUMNS)

    @pytest.mark.parametrize("model", [SIRFV])
    def test_deprecated(self, model):
        with pytest.raises(NotImplementedError):
            model(1, 1, 1, 1, 1)

    @pytest.mark.parametrize("model", [SEWIRF])
    def test_validation_deprecated(self, model):
        # Setting
        validator = ModelValidator(n_trials=1, seed=1)
        # Execute validation
        with pytest.raises(NotImplementedError):
            validator.run(model, timeout=10)
