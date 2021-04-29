#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import PhaseTracker, Term, SIRF


class TestPhaseTracker(object):
    @pytest.mark.parametrize("country", ["Japan"])
    def test_edit_phases(self, jhu_data, population_data, country):
        population = population_data.value(country=country)
        records_df, _ = jhu_data.records(
            country=country, start_date="01May2020", population=population)
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
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan), (01Feb, 28Feb),
        # (01Apr, 15Apr)
        tracker.deactivate(start="01Mar2021", end="31Mar2021")
        # Remove a phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan), (01Feb, 28Feb)
        tracker.remove_phase(start="01Apr2021", end="15Apr2021")
        # Tracking
        assert set(Term.SUB_COLUMNS).issubset(tracker.track().columns)
        # Check summary
        summary_df = tracker.summary()
        expected_df = pd.DataFrame(
            {
                Term.TENSE: [Term.PAST, Term.PAST, Term.PAST, Term.FUTURE, Term.FUTURE],
                Term.START: ["01May2020", "01Jun2020", "01Oct2020", "01Jan2021", "01Feb2021"],
                Term.END: ["31May2020", "30Sep2020", "31Dec2020", "31Jan2021", "28Feb2021"],
                Term.N: population,
            },
            index=["0th", "1st", "2nd", "3rd", "4th"],
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
        tracker.set_ode(model, param_df, tau)
        # Check summary
        df = tracker.summary()
        assert df.columns.tolist() == [
            Term.TENSE, Term.START, Term.END, Term.N, Term.ODE, Term.RT, *model.PARAMETERS,
            Term.TAU, *model.DAY_PARAMETERS]
