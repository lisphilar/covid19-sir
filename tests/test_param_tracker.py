#!/usr/bin/env python
# -*- coding: utf-8 -*-


import warnings
import pytest
from covsirphy import SIRF, PhaseSeries, ParamTracker


@pytest.fixture(scope="module")
def param_tracker(jhu_data, population_data):
    population = population_data.value(country="Japan")
    record_df = jhu_data.subset(country="Japan", population=population)
    series = PhaseSeries("01Apr2020", "01Nov2020", population)
    return ParamTracker(
        record_df=record_df, phase_series=series, area="Japan"
    )


class TestParamTracker(object):

    def test_before_trend(self, param_tracker):
        with pytest.raises(ValueError):
            param_tracker.find_phase("01May2020")

    def test_trend(self, param_tracker):
        warnings.simplefilter("ignore", category=UserWarning)
        param_tracker.trend(show_figure=True)
        assert isinstance(param_tracker.change_dates(), list)

    def test_trend_find_phase(self, param_tracker):
        with pytest.raises(IndexError):
            param_tracker.find_phase("01Jan2000")
        param_tracker.find_phase("01May2020")

    @pytest.mark.parametrize(
        "date", ["01Apr2020", "01May2020", "01Jun2020", "01Jul2020"])
    def test_separate(self, param_tracker, date):
        close_dates = param_tracker.near_change_dates()
        if date in close_dates:
            with pytest.raises(ValueError):
                param_tracker.separate(date)
        else:
            param_tracker.separate(date)

    def test_past_phases(self, param_tracker):
        all_phases, all_units = param_tracker.past_phases()
        sel_phases, sel_units = param_tracker.past_phases(
            phases=["0th", "1st"])
        assert set(sel_phases) == set(["0th", "1st"])
        assert set(sel_phases).issubset(all_phases)
        assert set(sel_units).issubset(all_units)

    def test_before_estimate(self, param_tracker):
        with pytest.raises(NameError):
            param_tracker.simulate()

    def test_estimate(self, param_tracker):
        param_tracker.estimate(
            SIRF, phases=["1st"], timeout=5, timeout_iteration=5)
        param_tracker.estimate(SIRF, timeout=5, timeout_iteration=5)
        with pytest.raises(IndexError):
            param_tracker.estimate(SIRF, timeout=5, timeout_iteration=5)

    def test_simulate(self, param_tracker):
        param_tracker.simulate()

    def test_score(self, param_tracker):
        # Scores of all phases
        metrics_list = ["MAE", "MSE", "MSLE", "RMSE", "RMSLE"]
        for metrics in metrics_list:
            score = param_tracker.score(metrics=metrics)
            assert isinstance(score, float)
        # Scores of target phases
        param_tracker.score(metrics="RMSLE", phases=["0th", "2nd"])
        # Errors with arguments
        with pytest.raises(TypeError):
            param_tracker.score(variables="Infected")
        with pytest.raises(KeyError):
            param_tracker.score(variables=["Susceptible"])
        with pytest.raises(ValueError):
            param_tracker.score(metrics="Subjective evaluation")
