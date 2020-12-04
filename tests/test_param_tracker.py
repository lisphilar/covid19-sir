#!/usr/bin/env python
# -*- coding: utf-8 -*-


from covsirphy.phase.phase_unit import PhaseUnit
import warnings
import pytest
from covsirphy import SIRF, PhaseSeries, ParamTracker


@pytest.fixture(scope="module")
def tracker(jhu_data, population_data):
    population = population_data.value(country="Japan")
    record_df = jhu_data.subset(country="Japan", population=population)
    series = ParamTracker.create_series("01Apr2020", "01Nov2020", population)
    assert isinstance(series, PhaseSeries)
    return ParamTracker(
        record_df=record_df, phase_series=series, area="Japan"
    )


class TestParamTracker(object):
    def test_series(self, tracker):
        assert isinstance(tracker.series, PhaseSeries)
        assert len(tracker) == len(tracker.series)

    def test_add(self, tracker):
        tracker.add(days=10)
        tracker.add(end_date="01Jun2020")
        tracker.add(days=100)
        tracker.add()
        with pytest.raises(ValueError):
            tracker.add(end_date="01May2020")

    def test_delete(self, tracker):
        length = len(tracker)
        tracker.delete(phases=["1st", "last"])
        assert len(tracker) == length - 2

    def test_delete_all(self, tracker):
        assert tracker
        tracker.delete_all()
        assert not tracker

    def test_combine(self, tracker):
        tracker.add("01May2020")
        tracker.add("01Jun2020")
        tracker.add("01Jul2020")
        tracker.add("01Aug2020")
        tracker.combine(phases=["2nd", "last"])
        tracker.combine(phases=["0th", "1st"])

    def test_before_trend(self, tracker):
        tracker.delete_all()
        with pytest.raises(ValueError):
            tracker.find_phase("01May2020")

    def test_trend(self, tracker):
        warnings.simplefilter("ignore", category=UserWarning)
        tracker.trend(show_figure=True)
        assert isinstance(tracker.change_dates(), list)

    def test_trend_find_phase(self, tracker):
        with pytest.raises(IndexError):
            tracker.find_phase("01Jan2000")
        tracker.find_phase("01May2020")

    @pytest.mark.parametrize(
        "date", ["01Apr2020", "01May2020", "01Jun2020", "01Jul2020"])
    def test_separate(self, tracker, date):
        close_dates = tracker.near_change_dates()
        if date in close_dates:
            with pytest.raises(ValueError):
                tracker.separate(date)
        else:
            tracker.separate(date)

    def test_past_phases(self, tracker):
        all_phases, all_units = tracker.past_phases()
        sel_phases, sel_units = tracker.past_phases(
            phases=["0th", "1st"])
        assert set(sel_phases) == set(["0th", "1st"])
        assert set(sel_phases).issubset(all_phases)
        assert set(sel_units).issubset(all_units)

    def test_future_phases(self, tracker):
        tracker.trend()
        phases0, units0 = tracker.future_phases()
        assert len(phases0) == len(units0) == 0
        tracker.add(end_date="01Apr2021")
        phases, units = tracker.future_phases()
        assert len(phases) == len(units) == 1
        assert isinstance(units[0], PhaseUnit)
        tracker.delete(phases=phases)

    def test_before_estimate(self, tracker):
        with pytest.raises(NameError):
            tracker.simulate()

    def test_estimate(self, tracker):
        tracker.estimate(
            SIRF, phases=["1st"], timeout=5, timeout_iteration=5)
        tracker.estimate(SIRF, timeout=5, timeout_iteration=5)
        with pytest.raises(IndexError):
            tracker.estimate(SIRF, timeout=5, timeout_iteration=5)

    def test_simulate(self, tracker):
        tracker.simulate()

    def test_score(self, tracker):
        # Scores of all phases
        metrics_list = ["MAE", "MSE", "MSLE", "RMSE", "RMSLE"]
        for metrics in metrics_list:
            score = tracker.score(metrics=metrics)
            assert isinstance(score, float)
        # Scores of target phases
        tracker.score(metrics="RMSLE", phases=["0th", "2nd"])
        # Errors with arguments
        with pytest.raises(TypeError):
            tracker.score(variables="Infected")
        with pytest.raises(KeyError):
            tracker.score(variables=["Susceptible"])
        with pytest.raises(ValueError):
            tracker.score(metrics="Subjective evaluation")
