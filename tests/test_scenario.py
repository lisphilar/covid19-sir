#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
from covsirphy import Scenario
from covsirphy import Term, SIRF


class TestScenario(object):
    def test_records(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="Italy")
        record_df = scenario.records(show_figure=False)
        assert isinstance(record_df, pd.DataFrame)
        assert set(record_df.columns) == set(Term.NLOC_COLUMNS)

    def test_analysis(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="Italy")
        # S-R trend analysis
        scenario.trend(show_figure=False)
        # Parameter estimation of SIR-F model
        scenario.estimate(model=SIRF)
        # Prediction
        scenario.add(name="Main", days=100)
        scenario.simulate(name="Main", show_figure=False)
        scenario.param_history(targets=["Rt"], name="Main", show_figure=False)
        # New scenario
        sigma_new = scenario.get("sigma", phase="last") * 2
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        scenario.add_phase(name="New medicines", days=100, sigma=sigma_new)
        # Summarize scenarios
        summary_df = scenario.summary()
        assert isinstance(summary_df, pd.DataFrame)
        desc_df = scenario.describe()
        assert isinstance(desc_df, pd.DataFrame)

    def test_add_past_phases(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="India")
        scenario.first_date = "01Mar2020"
        scenario.last_date = "16Jul2020"
        # With trend analysis
        scenario.trend(set_phases=True)
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
