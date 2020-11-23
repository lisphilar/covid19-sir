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
        first_len = len(df)
        assert set(df.loc["3rd", :].tolist()) == set(
            [Term.PAST, "27May2020", "01Aug2020", 123998518])
        assert set(df.loc["4th", :].tolist()) == set(
            [Term.FUTURE, "02Aug2020", "01Sep2020", 123998518])
        # Disable/enable a phase
        series.disable("0th")
        assert "0th" not in series.to_dict()
        assert len(series) == first_len - 1
        assert len([unit for unit in series]) == first_len
        assert len([unit for unit in series if unit]) == first_len - 1
        series.enable("0th")
        assert "0th" in series.to_dict()
        assert len(series) == first_len
        # Clear future phases: 4th and 5th will be deleted
        series.clear(include_past=False)
        assert "4th" not in series.to_dict()
        assert len(series) == first_len - 2
        assert series
        # Clear all phases
        series.clear(include_past=True)
        assert len(series) == 0
        assert not series
        # Filling past phases: 0th
        series.add()
        assert len(series) == 1
        # Last phase
        assert series.unit(phase="last")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_trend(self, jhu_data, population_data, country):
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # Setting
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        # S-R trend analysis
        series.trend(sr_df)
        series.trend_show(sr_df=sr_df, area=None, filename=None)
        # Un-registered phase
        with pytest.raises(KeyError):
            series.unit("100th")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_add_phase_with_model(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        series.trend(sr_df)
        # Add future phase with model and tau
        series.add(end_date="01Sep2020", model=SIR, tau=360)
        series.add(end_date="01Oct2020")
        length = len(series)
        assert series.to_dict()[Term.num2str(length - 2)][Term.ODE] == SIR.NAME
        assert series.to_dict()[Term.num2str(length - 1)][Term.TAU] == 360
        series.add(end_date="01Nov2020", rho=0.006)
        series.add(end_date="01Dec2020", sigma=0.011)
        assert series.to_dict()[Term.num2str(length + 1)][Term.RT] == 0.55
        assert series.to_dict()[Term.num2str(length + 1)]["1/beta [day]"] == 41

    @pytest.mark.parametrize("country", ["Japan"])
    def test_delete_phase(self, jhu_data, population_data, country):
        # Setting
        population = population_data.value(country)
        sr_df = jhu_data.to_sr(country=country, population=population)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        series.trend(sr_df)
        first_len = len(series)
        # Deletion of 0th phase is the same as disabling 0th phase
        series.delete("0th")
        series.enable("0th")
        assert len(series) == first_len
        assert "5th" in series.to_dict()
        # Delete phase (not the last registered phase)
        new_second = PhaseUnit(
            series.unit("2nd").start_date,
            series.unit("3rd").end_date,
            series.unit("2nd").population)
        series.delete("3rd")
        assert len(series) == first_len - 1
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
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        series.add(end_date="01May2020")
        series.add(end_date="01Jun2020")
        series.add(end_date="01Jul2020")
        series.add()
        # Replace one old phase with one new phase
        unit_old = series.unit("0th")
        unit_new = PhaseUnit(
            unit_old.start_date, unit_old.end_date, population
        )
        unit_new.set_ode(tau=360)
        series.replace("0th", unit_new)
        assert series.unit("0th") == unit_new
        # Replace one old phase with two new phases

        unit_old = series.unit("1st")
        change_date = Term.date_change(unit_old.end_date, days=-7)
        unit_pre = PhaseUnit(
            unit_old.start_date, Term.yesterday(change_date), population)
        unit_pre.set_ode(tau=360)
        unit_fol = PhaseUnit(change_date, unit_old.end_date, population)
        unit_fol.set_ode(tau=360)
        series.replaces(phase="1st", new_list=[unit_pre, unit_fol])
        print(series.unit("1st"), unit_pre)
        assert series.unit("1st") == unit_pre
        assert series.unit("2nd") == unit_fol
        # TypeError of new_list
        with pytest.raises(TypeError):
            series.replaces(phase="3rd", new_list=[unit_pre, Term])
        # ValueError with tense
        with pytest.raises(ValueError):
            future_unit = PhaseUnit("01Sep2020", "01Dec2020", population)
            series.replaces(phase="3rd", new_list=[future_unit])
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
