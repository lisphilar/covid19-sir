#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import PhaseTracker


class TestPhaseTracker(object):
    @pytest.mark.parametrize("country", ["Japan"])
    def test_define_phases(self, jhu_data, population_data, country):
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
