#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import Estimator, ODESimulator, Term, ExampleData, Scenario
from covsirphy import SIR, SIRD, SIRF  # , SIRFV, SEWIRF


class TestODESimulator(object):
    @pytest.mark.parametrize(
        "model",
        # TODO: add SIRFV and SEWIRF to fix issue #18
        [SIR, SIRD, SIRF]
    )
    def test_ode(self, model):
        # Setting
        eg_tau = 1440
        # Simulation
        example_data = ExampleData(tau=eg_tau)
        example_data.add(model)
        nondim_df = example_data.non_dim(model)
        assert isinstance(nondim_df, pd.DataFrame)
        nondim_cols = [Term.TS, *list(model.VAR_DICT.keys())]
        assert set(nondim_df.columns) == set(nondim_cols)
        clean_df = example_data.cleaned()
        assert isinstance(clean_df, pd.DataFrame)
        assert set(clean_df.columns) == set(Term.COLUMNS)
        dim_df = example_data.subset(model)
        assert isinstance(dim_df, pd.DataFrame)
        assert set(dim_df.columns) == set(Term.NLOC_COLUMNS)
        # Estimation
        population = model.EXAMPLE["population"]
        estimator = Estimator(
            example_data, model=model, population=population,
            country=model.NAME, province=Term.UNKNOWN, tau=eg_tau
        )
        estimator.run()
        estimated_df = estimator.summary(name=model.NAME)
        assert isinstance(estimated_df, pd.DataFrame)
        estimator.history(show_figure=False)
        estimator.accuracy(show_figure=False)

    def test_ode_two_phases(self, population_data):
        # Setting
        eg_tau = 1440
        # Simulation
        example_data = ExampleData(tau=eg_tau)
        example_data.add(SIRF, step_n=30)
        example_data.add(SIRD, step_n=30)
        nondim_df = example_data.non_dim(SIRF)
        assert isinstance(nondim_df, pd.DataFrame)
        nondim_cols = [Term.TS, *list(SIRF.VAR_DICT.keys())]
        assert set(nondim_df.columns) == set(nondim_cols)
        clean_df = example_data.cleaned()
        assert isinstance(clean_df, pd.DataFrame)
        assert set(clean_df.columns) == set(Term.COLUMNS)
        dim_df = example_data.subset(SIRF)
        assert isinstance(dim_df, pd.DataFrame)
        assert set(dim_df.columns) == set(Term.NLOC_COLUMNS)
        # Scenario analysis
        population = SIRF.EXAMPLE["population"]
        population_data.update(population, country=SIRF.NAME)
        scenario = Scenario(example_data, population_data, country=SIRF.NAME)
        scenario.trend()

    @pytest.mark.parametrize("model", [SIR])
    def test_ode_with_dataframe(self, model):
        # Setting
        eg_tau = 1440
        start_date = "22Jan2020"
        # Simulation
        simulator = ODESimulator(country="Example", province=model.NAME)
        simulator.add(model=model, **model.EXAMPLE)
        simulator.run()
        dim_df = simulator.dim(tau=eg_tau, start_date=start_date)
        # Estimation
        population = model.EXAMPLE["population"]
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        estimator = Estimator(
            dim_df, model=model, population=population,
            country="Example", province=model.NAME, tau=eg_tau
        )
        estimator.run()
