#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import ExampleData, PopulationData, Term, Scenario
from covsirphy import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF


class TestODE(object):
    @pytest.mark.parametrize(
        "model",
        [SIR, SIRD, SIRF, SIRFV, SEWIRF])
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
    def test_usage_mistakes(self, model):
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

    @pytest.mark.parametrize(
        "model",
        # SIRFV, SEWIRF
        [SIR, SIRD, SIRF])
    def test_estimate(self, model):
        # Setting
        eg_tau = 1440
        area = {"country": "Full", "province": model.NAME}
        # Population
        population_data = PopulationData(filename=None)
        population_data.update(model.EXAMPLE["population"], **area)
        # Simulation
        example_data = ExampleData(tau=eg_tau, start_date="01Jan2020")
        example_data.add(model, **area)
        # Estimation
        snl = Scenario(example_data, population_data, **area)
        snl.add()
        snl.estimate(model)
