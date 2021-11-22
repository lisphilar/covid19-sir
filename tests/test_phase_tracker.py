#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import PhaseTracker, Term, SIRF, UnExecutedError, UnExpectedValueError


class TestPhaseTracker(object):
    @pytest.mark.parametrize("country", ["Japan"])
    def test_edit_phases(self, jhu_data, country):
        records_df, _ = jhu_data.records(country=country, start_date="01May2020")
        population = records_df.loc[records_df.index[0], [Term.C, Term.S]].sum()
        # Create tracker -> no phases
        tracker = PhaseTracker(data=records_df, today="31Dec2020", area=country)
        # Add two past phase
        # -> (01May, 31May), (01Jun, 30Sep)
        tracker.define_phase(start="01Jun2020", end="30Sep2020")
        # Add a past phase and a future phase (because over today)
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan)
        tracker.define_phase(start="01Oct2020", end="31Jan2021")
        # Add a future phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan), (01Feb, 28Feb)
        tracker.define_phase(start="01Feb2021", end="28Feb2021")
        # Add two future phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan), (01Feb, 28Feb),
        # (01Mar, 31Mar), (01Apr, 15Apr)
        tracker.define_phase(start="01Apr2021", end="15APr2021")
        # Deactivate a future phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan),
        # (deactivated: 01Feb, 28Feb), (01Mar, 31Mar), (01Apr, 15Apr)
        tracker.deactivate(start="01Feb2021", end="28Feb2021")
        # Remove a phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan),
        # (deactivated: 01Feb, 28Feb), (01Mar, 31Mar)
        tracker.remove_phase(start="01Apr2021", end="15Apr2021")
        # Tracking
        assert set(Term.SUB_COLUMNS).issubset(tracker.track().columns)
        # Check summary
        summary_df = tracker.summary()
        expected_df = pd.DataFrame(
            {
                Term.TENSE: [Term.PAST, Term.PAST, Term.PAST, Term.FUTURE, Term.FUTURE],
                Term.START: ["01May2020", "01Jun2020", "01Oct2020", "01Jan2021", "01Mar2021"],
                Term.END: ["31May2020", "30Sep2020", "31Dec2020", "31Jan2021", "31Mar2021"],
                Term.N: population,
            },
            index=["0th", "1st", "2nd", "3rd", "5th"],
        )
        expected_df[Term.START] = pd.to_datetime(expected_df[Term.START])
        expected_df[Term.END] = pd.to_datetime(expected_df[Term.END])
        assert summary_df.equals(expected_df)
        # Activate a phase
        tracker.define_phase(start="01Mar2021", end="31Mar2021")
        df = tracker.summary()
        assert df.loc[df.index[-1], Term.END] == pd.to_datetime("31Mar2021")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_trend(self, jhu_data, population_data, country, imgfile):
        population = population_data.value(country=country)
        records_df, _ = jhu_data.records(
            country=country, start_date="01May2020", population=population)
        # Create tracker -> no phases
        tracker = PhaseTracker(data=records_df, today="31Dec2020", area=country)
        # Define phases with S-R trend analysis
        tracker.trend(force=True, show_figure=False)
        # Show trend
        tracker.trend(force=False, show_figure=True, filename=imgfile)
        # Check summary
        df = tracker.summary()
        assert df.loc[df.index[-1], Term.END] == pd.to_datetime("31Dec2020")
        assert Term.FUTURE not in df[Term.TENSE]

    @pytest.mark.parametrize("country", ["Japan"])
    @pytest.mark.parametrize("model", [SIRF])
    @pytest.mark.parametrize("tau", [720, None])
    @pytest.mark.parametrize("metric", ["RMSLE"])
    def test_estimate(self, jhu_data, population_data, country, model, tau, metric):
        population = population_data.value(country=country)
        records_df, _ = jhu_data.records(
            country=country, start_date="01Nov2020", population=population)
        # Create tracker -> no phases
        tracker = PhaseTracker(data=records_df, today="31Dec2020", area=country)
        # Define phases with S-R trend analysis
        tracker.trend(force=True, show_figure=False)
        # Estimate tau value (if necessary) and parameter values
        with pytest.raises(UnExecutedError):
            tracker.simulate()
        est_tau = tracker.estimate(
            model=model, tau=tau, metric=metric, timeout=1, timeout_iteration=1)
        assert isinstance(est_tau, int)
        if tau is not None:
            assert est_tau == tau
        # Check summary
        df = tracker.summary()
        assert df.columns.tolist() == [
            Term.TENSE, Term.START, Term.END, Term.N, Term.ODE, Term.RT, *model.PARAMETERS,
            Term.TAU, *model.DAY_PARAMETERS, metric, Term.TRIALS, Term.RUNTIME]
        # Simulation
        sim_df = tracker.simulate()
        assert sim_df.columns.tolist() == Term.SUB_COLUMNS
        # Add/track a future phase
        tracker.define_phase(start="01Jan2021", end="31Jan2021")
        track_df = tracker.track()
        assert not track_df[Term.SUB_COLUMNS].isna().sum().sum()

    @pytest.mark.parametrize("country", ["Japan"])
    @pytest.mark.parametrize("model", [SIRF])
    @pytest.mark.parametrize("tau", [720])
    def test_set_ode(self, jhu_data, population_data, country, model, tau):
        population = population_data.value(country=country)
        records_df, _ = jhu_data.records(
            country=country, start_date="01Nov2020", population=population)
        # Create tracker -> no phases
        tracker = PhaseTracker(data=records_df, today="31Dec2020", area=country)
        # Define phases with S-R trend analysis
        tracker.trend(force=True, show_figure=False)
        # Set ODE, tau and parameter values
        param_dict = model.EXAMPLE[Term.PARAM_DICT].copy()
        param_df = pd.DataFrame(index=records_df[Term.DATE])
        for (param, value) in param_dict.items():
            param_df[param] = value
        with pytest.raises(UnExecutedError):
            tracker.simulate()
        tracker.set_ode(model, param_df, tau)
        # Check summary
        df = tracker.summary()
        assert df.columns.tolist() == [
            Term.TENSE, Term.START, Term.END, Term.N, Term.ODE, Term.RT, *model.PARAMETERS,
            Term.TAU, *model.DAY_PARAMETERS]
        # Simulation
        sim_df = tracker.simulate()
        assert sim_df.columns.tolist() == Term.SUB_COLUMNS
        # Add/track a future phase
        tracker.define_phase(start="01Jan2021", end="31Jan2021")
        track_df = tracker.track()
        assert not track_df[Term.SUB_COLUMNS].isna().sum().sum()

    @pytest.mark.parametrize("country", ["Japan"])
    def test_parse_range(self, jhu_data, population_data, country):
        population = population_data.value(country=country)
        records_df, _ = jhu_data.records(
            country=country, start_date="01May2020", population=population)
        # Create tracker -> no phases
        today = pd.to_datetime("31Dec2020")
        tracker = PhaseTracker(data=records_df, today=today, area=country)
        with pytest.raises(UnExecutedError):
            tracker.parse_range()
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), today, (01Jan2021, 31Jan)
        tracker.define_phase(start="01Jun2020", end="30Sep2020")
        tracker.define_phase(start="01Oct2020", end="31Jan2021")
        first, last = pd.to_datetime("01May2020"), pd.to_datetime("31Jan2021")
        # With @date argument
        with pytest.raises(ValueError):
            tracker.parse_range(dates=(1, 2, 3))
        start, end = pd.to_datetime("15May2020"), pd.to_datetime("15Nov2020")
        assert tracker.parse_range(dates=(start, end)) == (start, end)
        assert tracker.parse_range(dates=(start, None)) == (start, last)
        assert tracker.parse_range(dates=(None, end)) == (first, end)
        # With @past_days argument
        assert tracker.parse_range(past_days=15) == (pd.to_datetime("16Dec2020"), today)
        # With @phases argument
        with pytest.raises(UnExpectedValueError):
            tracker.parse_range(phases=["5th"])
        assert tracker.parse_range(phases=["1st", "3rd"]) == (
            pd.to_datetime("01Jun2020"), pd.to_datetime("31Jan2021"))
        assert tracker.parse_range(phases=["last"]) == (
            pd.to_datetime("01Jan2021"), pd.to_datetime("31Jan2021"))
        # No arguments were specified
        assert tracker.parse_range() == (first, last)
