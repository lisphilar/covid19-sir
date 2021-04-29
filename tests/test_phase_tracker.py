#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import PhaseTracker, Term


class TestPhaseTracker(object):
    @pytest.mark.parametrize("country", ["Japan"])
    def test_edit_phases(self, jhu_data, population_data, country):
        population = population_data.value(country=country)
        records_df, _ = jhu_data.records(
            country=country, start_date="01May2020", population=population)
        # Create tracker -> no phases
        tracker = PhaseTracker(data=records_df, today="31Dec2020")
        # Add two past phase
        # -> (01May, 31May), (01Jun, 30Sep)
        tracker.define_phase(start="01Jun2020", end="30Sep2020")
        # Add a past phase and a future phase (because over today)
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan)
        tracker.define_phase(start="01Oct2020", end="31Jan2021")
        # Add a future phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan), (01Feb, 28Feb)
        tracker.define_phase(start="01Feb2021", end="28Feb2021")
        # Add a future phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan), (01Feb, 28Feb), (01Mar, 31Mar)
        tracker.define_phase(start="01Mar2021", end="31Mar2021")
        # Remove a future phase
        # -> (01May, 31May), (01Jun, 30Sep), (01Oct, 31Dec), (01Jan2021, 31Jan), (01Feb, 28Feb)
        tracker.remove_phase(start="01Mar2021", end="31Mar2021")
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
