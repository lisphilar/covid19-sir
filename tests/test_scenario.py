#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import Scenario
from covsirphy import Term, PopulationData, SIRF, SIRFV, ExampleData


class TestScenario(object):
    def test_records(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="Italy")
        record_df = scenario.records(show_figure=False)
        warnings.filterwarnings("ignore", category=UserWarning)
        record_df = scenario.records(show_figure=True)
        assert isinstance(record_df, pd.DataFrame)
        assert set(record_df.columns) == set(Term.NLOC_COLUMNS)

    def test_analysis(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="Italy")
        with pytest.raises(ValueError):
            scenario.estimate(model=SIRF)
        # S-R trend analysis
        scenario.trend(show_figure=False)
        warnings.filterwarnings("ignore", category=UserWarning)
        scenario.trend(show_figure=True)
        # Parameter estimation of SIR-F model
        with pytest.raises(ValueError):
            scenario.param_history(targets=["Rt"], show_figure=False)
        with pytest.raises(ValueError):
            scenario.estimate(model=SIRF, tau=1440)
        scenario.estimate(model=SIRF)
        # History of estimation
        scenario.estimate_history(phase="1st")
        with pytest.raises(KeyError):
            scenario.estimate_history(phase="0th")
        # Accuracy of estimation
        scenario.estimate_accuracy(phase="1st")
        with pytest.raises(KeyError):
            scenario.estimate_accuracy(phase="0th")
        # Simulation
        scenario.add(name="Main", days=100)
        scenario.simulate(name="Main", show_figure=False)
        scenario.simulate(name="Main", show_figure=True)
        scenario.param_history(targets=["Rt"], show_figure=False)
        scenario.param_history(targets=["Rt"], divide_by_first=False)
        scenario.param_history(targets=["Rt"], show_box_plot=False)
        with pytest.raises(KeyError):
            scenario.param_history(targets=["Rt", "Value"])
        with pytest.raises(KeyError):
            scenario.param_history(targets=["Rt"], box_plot=False)
        # New scenario
        sigma_new = scenario.get("sigma", phase="last") * 2
        with pytest.raises(KeyError):
            scenario.get("value")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        scenario.add_phase(name="New medicines", days=100, sigma=sigma_new)
        # Summarize scenarios
        summary_df = scenario.summary()
        assert isinstance(summary_df, pd.DataFrame)
        desc_df = scenario.describe()
        assert isinstance(desc_df, pd.DataFrame)
        # Estimation errors
        with pytest.raises(TypeError):
            scenario.estimate(SIRF, phases="1st")
        with pytest.raises(KeyError):
            scenario.estimate(SIRF, phases=["100th"])

    def test_add_past_phases(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="India")
        scenario.delete()
        # Phase series
        scenario.clear(name="Medicine")
        scenario.add(days=100, name="Medicine")
        scenario.delete(name="Medicine")
        with pytest.raises(TypeError):
            scenario.delete(phase="0th")
        with pytest.raises(TypeError):
            scenario.summary(columns="Population")
        with pytest.raises(KeyError):
            scenario.summary(columns=["Value"])
        # Range of past phases
        scenario.first_date = "01Mar2020"
        assert scenario.first_date == "01Mar2020"
        scenario.last_date = "16Jul2020"
        assert scenario.last_date == "16Jul2020"
        with pytest.raises(ValueError):
            scenario.first_date = "01Aug2020"
        with pytest.raises(ValueError):
            scenario.last_date = "01Feb2020"
        # With trend analysis
        scenario.trend(set_phases=True)
        with pytest.raises(ValueError):
            scenario.trend(set_phases=False, n_points=3)
        scenario.combine(phases=["3rd", "4th"])
        scenario.separate(date="30May2020", phase="1st")
        scenario.delete(phases=["1st"])
        scenario.trend(set_phases=False)
        trend_df = scenario.summary()
        assert len(trend_df) == 4
        # add scenarios one by one
        scenario.clear(include_past=True)
        scenario.add(end_date="29May2020")
        scenario.add(end_date="05Jun2020").delete(phases=["0th"])
        scenario.add(end_date="15Jun2020")
        scenario.add(end_date="04Jul2020")
        scenario.add()
        one_df = scenario.summary()
        assert len(one_df) == 4

    def test_scenario_with_model_change(self):
        # Instance to save population values
        population_data = PopulationData(filename=None)
        # Set tau value and start date of records
        example_data = ExampleData(tau=1440, start_date="01Jan2020")
        # Preset of SIR-F parameters
        preset_dict = SIRF.EXAMPLE["param_dict"]
        # Create dataset from 01Jan2020 to 31Jan2020
        area = {"country": "Theoretical"}
        example_data.add(SIRF, step_n=30, **area)
        # Register population value
        population_data.update(SIRF.EXAMPLE["population"], **area)
        # Create Scenario class
        snl = Scenario(example_data, population_data, tau=1440, **area)
        # Set 0th phase from 02Jan2020 to 31Jan2020 with preset parameter values
        snl.clear(include_past=True)
        snl.add(end_date="31Jan2020", model=SIRF, **preset_dict)
        # Add main scenario
        snl.add(end_date="31Dec2020", name="Main")
        # Add lockdown scenario
        rho_lock = snl.get("rho", phase="0th") / 2
        snl.add(end_date="31Dec2020", name="Lockdown", rho=rho_lock)
        # Add medicine scenario
        kappa_med = snl.get("kappa", phase="0th") / 2
        sigma_med = snl.get("sigma", phase="0th") * 2
        snl.add(
            end_date="31Dec2020", name="Medicine", kappa=kappa_med, sigma=sigma_med)
        # Add vaccine scenario
        snl.add(
            end_date="31Dec2020", name="Vaccine", model=SIRFV, omega=0.001)
        # Summarize
        snl.summary()
        # Compare scenarios
        snl.describe(y0_dict={"Vaccinated": 0})
