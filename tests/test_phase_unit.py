#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import PhaseUnit
from covsirphy import Term, SIR, Estimator


class TestPhaseUnit(object):
    def test_start(self):
        unit = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        assert str(unit) == "Phase (01Jan2020 - 01Feb2020)"
        summary_dict = unit.to_dict()
        assert summary_dict[Term.START] == "01Jan2020"
        assert summary_dict[Term.END] == "01Feb2020"
        assert summary_dict[Term.N] == 1000
        summary_df = unit.summary()
        assert set(summary_df.columns) == set([Term.START, Term.END, Term.N])
        with pytest.raises(NotImplementedError):
            assert unit == "phase"
        unit2 = PhaseUnit("01Jan2020", "01Feb2020", 100000)
        assert unit == unit2
        unit3 = PhaseUnit("01Jan2020", "01Mar2020", 1000)
        assert unit != unit3
        assert "31Dec2019" not in unit
        assert "01Jan2020" in unit
        assert "15Jan2020" in unit
        assert "01Feb2020" in unit
        assert "01Mar2020" not in unit

    def test_sort(self):
        unit1 = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        unit2 = PhaseUnit("02Feb2020", "01Mar2020", 1000)
        unit3 = PhaseUnit("02Mar2020", "01Apr2020", 1000)
        unit4 = PhaseUnit("02Mar2020", "01Apr2020", 1000)
        unit5 = PhaseUnit("01Jan2020", "01Apr2020", 1000)
        assert unit3 == unit4
        assert unit1 != unit2
        assert unit1 < unit2
        assert unit3 > unit1
        assert not unit3 > unit4
        assert unit3 <= unit4
        assert unit1 < "02Feb2020"
        assert unit1 <= "01Feb2020"
        assert unit1 > "31Dec2019"
        assert unit1 >= "01Jan2020"
        assert sorted([unit3, unit1, unit2]) == [unit1, unit2, unit3]
        assert str(unit1 + unit2) == "Phase (01Jan2020 - 01Mar2020)"
        assert str(unit5 - unit1) == "Phase (02Feb2020 - 01Apr2020)"
        assert str(unit5 - unit4) == "Phase (01Jan2020 - 01Mar2020)"
        assert set([unit1, unit3, unit4]) == set([unit1, unit3])

    def test_enable(self):
        unit = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        assert bool(unit)
        unit.disable()
        assert not bool(unit)
        unit.enable()
        assert bool(unit)

    def test_definition_property(self):
        unit = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        assert unit.start_date == "01Jan2020"
        with pytest.raises(AttributeError):
            unit.start_date = "10Jan2020"
        assert unit.end_date == "01Feb2020"
        with pytest.raises(AttributeError):
            unit.end_date = "10Jan2020"
        assert unit.population == 1000
        with pytest.raises(AttributeError):
            unit.population = 100000

    def test_tau(self):
        unit = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        assert unit.tau is None
        with pytest.raises(ValueError):
            unit.tau = 1000
        unit.tau = 1440
        assert unit.tau == 1440
        with pytest.raises(AttributeError):
            unit.tau = 720

    def test_set_ode(self):
        unit = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        unit.set_ode(model=SIR, tau=720, rho=0.2)
        assert issubclass(unit.model, SIR)
        summary_dict = unit.to_dict()
        assert summary_dict[Term.TAU] == 720
        assert summary_dict[Term.ODE] == SIR.NAME
        assert summary_dict["rho"] == 0.2
        assert summary_dict["sigma"] is None

    def test_set_tau(self):
        unit = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        unit.set_ode(model=None, tau=240, rho=0.006)
        assert unit.tau == 240
        assert "rho" not in unit.to_dict()

    @pytest.mark.parametrize("country", ["Japan"])
    def test_estimate(self, jhu_data, population_data, country):
        # Dataset
        population = population_data.value(country)
        record_df = jhu_data.subset(country, population=population)
        # Parameter estimation
        unit = PhaseUnit("27May2020", "27Jun2020", population)
        with pytest.raises(NameError):
            unit.estimate()
        unit.set_ode(model=SIR)
        with pytest.raises(ValueError):
            unit.estimate()
        unit.record_df = record_df
        assert set(unit.record_df.columns) == set(Term.SUB_COLUMNS)
        unit.estimate()
        # Check results
        assert isinstance(unit.estimator, Estimator)
        assert set(SIR.PARAMETERS).issubset(unit.to_dict())
        assert set(SIR.DAY_PARAMETERS).issubset(unit.to_dict())
        cols = [Term.ODE, Term.RMSLE, Term.TRIALS, Term.RUNTIME]
        assert set(cols).issubset(unit.to_dict())
        assert None not in unit.to_dict().values()

    @pytest.mark.parametrize("country", ["Japan"])
    def test_estimate_with_fixed(self, jhu_data, population_data, country):
        # Dataset
        population = population_data.value(country)
        record_df = jhu_data.subset(country, population=population)
        # Setting
        unit = PhaseUnit("27May2020", "27Jun2020", population)
        unit.set_ode(model=SIR, tau=360, rho=0.01)
        # Parameter estimation
        unit.estimate(record_df=record_df)
        assert unit.tau == unit.to_dict()[Term.TAU] == 360
        assert unit.to_dict()["rho"] == 0.01

    @pytest.mark.parametrize("country", ["Japan"])
    def test_simulate(self, jhu_data, population_data, country):
        # Dataset
        population = population_data.value(country)
        record_df = jhu_data.subset(country, population=population)
        # Parameter setting
        unit = PhaseUnit("27May2020", "27Jun2020", population)
        with pytest.raises(NameError):
            unit.simulate()
        with pytest.raises(ValueError):
            unit2 = PhaseUnit("27May2020", "27Jun2020", population)
            unit2.set_ode(model=SIR)
            unit2.simulate()
        with pytest.raises(ValueError):
            unit3 = PhaseUnit("27May2020", "27Jun2020", population)
            unit3.set_ode(model=SIR, tau=240, rho=0.006)
            unit3.simulate()
        unit.set_ode(model=SIR, tau=240, rho=0.006, sigma=0.011)
        summary_dict = unit.to_dict()
        assert summary_dict[Term.RT] == 0.55
        assert summary_dict["1/beta [day]"] == 27
        assert summary_dict["1/gamma [day]"] == 15
        with pytest.raises(ValueError):
            unit.simulate()
        # Set initial values
        unit.set_y0(record_df)
        # Simulation
        sim_df = unit.simulate()
        assert set(sim_df.columns) == set(Term.NLOC_COLUMNS)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_simulate_y0_direct(self, jhu_data, population_data, country):
        # Dataset
        population = population_data.value(country)
        # Parameter setting
        unit = PhaseUnit("27May2020", "27Jun2020", population)
        unit.set_ode(model=SIR, tau=240, rho=0.006, sigma=0.011)
        # Simulation
        y0_dict = {
            "Susceptible": 126512455, "Infected": 1806, "Fatal or Recovered": 14839
        }
        sim_df = unit.simulate(y0_dict=y0_dict)
        assert set(sim_df.columns) == set(Term.NLOC_COLUMNS)
