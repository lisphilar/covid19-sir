#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy import Scenario
from covsirphy import Word, SIRF


class TestScenario(object):
    def test_records(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="Italy")
        record_df = scenario.records(show_figure=False)
        assert isinstance(record_df, pd.DataFrame)
        assert set(record_df.columns) == set(Word.NLOC_COLUMNS)

    def test_analysis(self, jhu_data, population_data):
        scenario = Scenario(jhu_data, population_data, country="Italy")
        # S-R trend analysis
        scenario.trend(show_figure=False)
        # Parameter estimation of SIR-F model
        scenario.estimate(model=SIRF)
        # Prediction
        scenario.add_phase(name="Main", days=100)
        scenario.simulate(name="Main", show_figure=False)
        scenario.param_history(targets=["Rt"], name="Main", show_figure=False)
        # New scenario
        sigma_new = scenario.get("sigma", phase="last") * 2
        scenario.add_phase(name="New medicines", days=100, sigma=sigma_new)
        # Summarize scenarios
        summary_df = scenario.summary()
        assert isinstance(summary_df, pd.DataFrame)
        desc_df = scenario.describe()
        assert isinstance(desc_df, pd.DataFrame)
