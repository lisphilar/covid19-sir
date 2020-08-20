#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import PhaseSeries
from covsirphy import Term, PhaseUnit, SIR


class TestPhaseSeries(object):

    @pytest.mark.parametrize("country", ["Japan"])
    def test_start(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        series = PhaseSeries("01Apr2020", "22Apr2020", population)
        # Whether units are registered or not
        assert not series
        assert series.summary().empty

    @pytest.mark.parametrize("country", ["Japan"])
    def test_add_phase(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        # Last phase when empty
        empty_phase = PhaseUnit("31Mar2020", "31Mar2020", population)
        assert series.unit(phase="last") == empty_phase
        # Add a phase with specified end date: 0th
        series.add(end_date="22Apr2020")
        # Add a phase with specified population value: 1st
        with pytest.raises(ValueError):
            series.add(end_date="22Apr2020")
        series.add(end_date="05May2020", population=int(population * 0.98))
        # Add a phase with specified the number of days: 2nd
        series.add(days=21)
        # Filling past phases and add a future phase: 3rd, 4th
        series.add(end_date="01Sep2020")
        # Add a future phase: 5th
        series.add(days=30)
        # Summary
        df = series.summary()
        base_cols = [Term.TENSE, Term.START, Term.END, Term.N]
        assert set(df.columns) == set(base_cols)
        assert series.to_dict()["0th"]["Type"] == Term.PAST
        assert len(df) == 6
        assert set(df.loc["3rd", :].tolist()) == set(
            [Term.PAST, "28May2020", "01Aug2020", 123998518])
        assert set(df.loc["4th", :].tolist()) == set(
            [Term.FUTURE, "02Aug2020", "01Sep2020", 123998518])
        # Disable/enable a phase
        series.disable("0th")
        assert "0th" not in series.to_dict()
        assert len(series) == 5
        assert len([unit for unit in series]) == 6
        assert len([unit for unit in series if unit]) == 5
        series.enable("0th")
        assert "0th" in series.to_dict()
        assert len(series) == 6
        # Clear future phases: 4th and 5th will be deleted
        series.clear(include_past=False)
        assert "4th" not in series.to_dict()
        assert len(series) == 4
        assert series
        # Clear all phases
        series.clear(include_past=True)
        assert len(series) == 0
        assert not series
        # Filling past phases: 0th
        series.add()
        assert len(series) == 1

    @pytest.mark.parametrize("country", ["Japan"])
    def test_trend(self, jhu_data, population_data, country):
        warnings.filterwarnings("ignore", category=UserWarning)
        # Setting
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        # Add a phase with specified end date
        series.add(end_date="22Apr2020")
        # S-R trend analysis
        series.trend(sr_df, set_phases=False, area=None)
        series.trend(sr_df, set_phases=False, area=None, show_figure=False)
        assert len(series) == 1
        assert set(series.to_dict()) == set(["0th"])
        # S-R trend analysis and set phases
        series.trend(sr_df)
        series.trend(sr_df, show_figure=False)
        # Summary
        assert not series.unit("0th")
        assert len(series) == 5
        # Last phase
        last_phase = PhaseUnit("13Jul2020", "01Aug2020", population)
        assert series.unit(phase="last") == last_phase
        # 3rd phase
        third_phase = PhaseUnit("27May2020", "27Jun2020", population)
        assert series.unit(phase="3rd") == third_phase
        # Un-registered phase
        with pytest.raises(KeyError):
            series.unit("10th")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_add_phase_with_model(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        series.trend(sr_df, show_figure=False)
        # Add future phase with model and tau
        series.add(end_date="01Sep2020", model=SIR, tau=360)
        series.add(end_date="01Oct2020")
        assert series.to_dict()["6th"][Term.ODE] == SIR.NAME
        assert series.to_dict()["7th"][Term.TAU] == 360
        series.add(end_date="01Nov2020", rho=0.006)
        series.add(end_date="01Dec2020", sigma=0.011)
        assert series.to_dict()["9th"][Term.RT] == 0.55
        assert series.to_dict()["9th"]["1/beta [day]"] == 41

    @pytest.mark.parametrize("country", ["Japan"])
    def test_delete_phase(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        series.trend(sr_df, show_figure=False)
        assert len(series) == 5
        # Deletion of 0th phase is the same as disabling 0th phase
        series.enable("0th")
        series.delete("0th")
        assert len(series) == 5
        assert "5th" in series.to_dict()
        assert not series.unit("0th")
        # Delete phase (not the last registered phase)
        new_second = PhaseUnit(
            series.unit("2nd").start_date,
            series.unit("3rd").end_date,
            series.unit("2nd").population)
        series.delete("3rd")
        assert len(series) == 4
        assert series.unit("2nd") == new_second
        # Delete the last phase
        old_last = series.unit("last")
        series.delete("last")
        series.add()
        assert series.unit("last").start_date == old_last.start_date

    @pytest.mark.parametrize("country", ["Japan"])
    def test_replace(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        series.trend(sr_df, show_figure=False)
        # Replace one old phase with one new phase
        unit_old = series.unit("2nd")
        unit_new = PhaseUnit(
            unit_old.start_date, unit_old.end_date, population
        )
        unit_new.set_ode(tau=360)
        series.replace("2nd", unit_new)
        assert series.unit("2nd") == unit_new
        # Replace one old phase with two new phases
        unit_old = series.unit("2nd")
        change_date = Term.date_change(unit_old.end_date, days=-7)
        unit_pre = PhaseUnit(
            unit_old.start_date, Term.yesterday(change_date), population)
        unit_pre.set_ode(tau=360)
        unit_fol = PhaseUnit(change_date, unit_old.end_date, population)
        unit_fol.set_ode(tau=360)
        series.replaces(phase="2nd", new_list=[unit_pre, unit_fol])
        print(series.unit("2nd"), unit_pre)
        assert series.unit("2nd") == unit_pre
        assert series.unit("3rd") == unit_fol
        # TypeError of new_list
        with pytest.raises(TypeError):
            series.replaces(phase="2nd", new_list=[unit_pre, Term])
        # ValueError with tense
        with pytest.raises(ValueError):
            future_unit = PhaseUnit("01Sep2020", "01Dec2020", population)
            series.replaces(phase="2nd", new_list=[future_unit])
        # Add phase without deletion of any phases
        new1 = PhaseUnit("02Aug2020", "01Sep2020", population)
        new2 = PhaseUnit("02Sep2020", "01Oct2020", population)
        series.replaces(phase=None, new_list=[new1, new2])
        assert series.unit("last") == new2

    @pytest.mark.parametrize("country", ["Japan"])
    def test_simulate(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        record_df = jhu_data.subset(country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        # Simulation
        series.add(
            end_date="22Apr2020", model=SIR, tau=360, rho=0.006, sigma=0.011)
        df = series.simulate(record_df=record_df)
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
