#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import warnings
import pandas as pd
import pytest
from covsirphy import ScenarioNotFoundError
from covsirphy import Scenario, DataHandler
from covsirphy import Term, PhaseSeries, SIR, SIRF


class TestDataHandler(object):
    @pytest.mark.parametrize("country", ["Italy", "Japan", "Netherlands", "Greece"])
    @pytest.mark.parametrize("province", [None, "Abruzzo"])
    def test_start(self, jhu_data, population_data, country, province):
        if province == "Abruzzo" and country != "Italy":
            with pytest.raises(KeyError):
                DataHandler(
                    jhu_data, population_data, country, province=province)
            return
        DataHandler(jhu_data, population_data, country, province=province)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_start_record_range(self, jhu_data, population_data, country):
        # Setting
        dhl = DataHandler(jhu_data, population_data, country)
        dhl.init_records()
        # Test
        dhl.first_date = "01Apr2020"
        assert dhl.first_date == "01Apr2020"
        dhl.last_date = "01May2020"
        assert dhl.last_date == "01May2020"
        with pytest.raises(ValueError):
            dhl.first_date = "01Jan2019"
        with pytest.raises(ValueError):
            tomorrow = Term.tomorrow(datetime.now().strftime(Term.DATE_FORMAT))
            dhl.last_date = tomorrow

    @pytest.mark.parametrize("country", ["Japan"])
    def test_records(self, jhu_data, population_data, country):
        warnings.simplefilter("ignore", category=UserWarning)
        # Setting
        dhl = DataHandler(jhu_data, population_data, country)
        dhl.init_records()
        dhl.first_date = "01Apr2020"
        dhl.last_date = "01Aug2020"
        # Test
        df = dhl.records(show_figure=False)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
        dates = df[Term.DATE]
        assert dates.min() == Term.date_obj(dhl.first_date)
        assert dates.max() == Term.date_obj(dhl.last_date)
        df2 = dhl.records(show_figure=True)
        assert isinstance(df2, pd.DataFrame)
        assert set(df2.columns) == set(Term.NLOC_COLUMNS)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_records_diff(self, jhu_data, population_data, country):
        warnings.simplefilter("ignore", category=UserWarning)
        dhl = DataHandler(jhu_data, population_data, country)
        dhl.init_records()
        dhl.records_diff(window=7, show_figure=False)
        dhl.records_diff(window=100, show_figure=True)


class TestScenario(object):
    @pytest.mark.parametrize("country", ["Japan"])
    def test_class_as_dict(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        # Create a phase series
        population = population_data.value(country)
        series = PhaseSeries("01Apr2020", "01Aug2020", population)
        with pytest.raises(ScenarioNotFoundError):
            assert snl["New"]
        # Add scenario
        snl["New"] = series
        # Get scenario
        assert snl["New"] == series

    @pytest.mark.parametrize("country", ["Japan"])
    def test_edit_series(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        assert snl.first_date == "01Apr2020"
        assert snl.last_date == "01Aug2020"
        # Add and clear
        assert snl.summary().empty
        snl.add(end_date="05May2020")
        with pytest.raises(ValueError):
            snl.add(end_date="20Apr2020")
        snl.add(days=20)
        snl.add()
        snl.add(end_date="01Sep2020")
        assert len(snl["Main"]) == 4
        snl.clear(include_past=True)
        with pytest.raises(ScenarioNotFoundError):
            snl.clear("New", template="Un-registered")
        snl.add(end_date="01Sep2020", name="New")
        assert len(snl["New"]) == 2
        # Delete
        snl.delete(name="Main")
        assert len(snl["Main"]) == 0
        with pytest.raises(TypeError):
            snl.delete(phases="1st", name="New")
        snl.delete(phases=["1st"], name="New")
        assert len(snl["New"]) == 1
        snl.delete(name="New")
        with pytest.raises(KeyError):
            assert len(snl["New"]) == 1

    @pytest.mark.parametrize("country", ["Japan"])
    def test_add_phase_dep(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        # Test
        warnings.simplefilter("error")
        with pytest.raises(DeprecationWarning):
            snl.add_phase(end_date="01May2020")
        warnings.simplefilter("ignore")
        snl.add_phase(end_date="01May2020")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_trend(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        # Deprecated
        warnings.simplefilter("error")
        with pytest.raises(DeprecationWarning):
            snl.trend(include_init_phase=False, show_figure=False)
        warnings.simplefilter("ignore")
        snl.trend(include_init_phase=False, show_figure=False)
        # S-R trend analysis
        snl.trend(show_figure=False)
        assert snl["Main"]
        with pytest.raises(ValueError):
            snl.trend(show_figure=False, n_points=3)
        # Disable/enable
        length = len(snl["Main"])
        snl.disable(phases=["0th"], name="Main")
        assert len(snl["Main"]) == length - 1
        snl.enable(phases=["0th"], name="Main")
        assert len(snl["Main"]) == length
        with pytest.raises(TypeError):
            snl.enable(phases="0th", name="Main")
        with pytest.raises(TypeError):
            snl.disable(phases="1st", name="Main")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_edit(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        snl.trend(show_figure=False)
        # Combine
        length = len(snl["Main"])
        snl.combine(["1st", "2nd"])
        n_changed = int(population_data.value(country) * 0.98)
        snl.combine(["2nd", "3rd"], population=n_changed)
        assert len(snl["Main"]) == length - 2
        # Separate
        snl.separate(date="01May2020")
        assert len(snl["Main"]) == length - 1

    @pytest.mark.parametrize("country", ["Japan"])
    def test_summary(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        snl.trend(show_figure=False)
        # One scenario
        assert set(snl.summary().columns) == set(
            [Term.TENSE, Term.START, Term.END, Term.N])
        # Show two scenarios
        snl.clear(name="New")
        cols = snl.summary().reset_index().columns
        assert set([Term.SERIES, Term.PHASE]).issubset(cols)
        # Show selected scenario
        cols_sel = snl.summary(name="New").reset_index().columns
        assert not set([Term.SERIES, Term.PHASE]).issubset(cols_sel)
        # Columns to show
        show_cols = [Term.N, Term.START]
        assert set(snl.summary(columns=show_cols).columns) == set(show_cols)
        with pytest.raises(TypeError):
            snl.summary(columns=Term.N)
        with pytest.raises(KeyError):
            snl.summary(columns=[Term.N, "Temperature"])
        # To markdown
        snl.summary().to_markdown()

    @pytest.mark.parametrize("country", ["Japan"])
    def test_estimate(self, jhu_data, population_data, country):
        warnings.simplefilter("ignore", category=UserWarning)
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        with pytest.raises(ValueError):
            snl.estimate(SIR)
        snl.trend(show_figure=False)
        with pytest.raises(AttributeError):
            snl.estimate_history(phase="last")
        # Parameter estimation
        with pytest.raises(KeyError):
            snl.estimate(SIR, phases=["30th"])
        with pytest.raises(ValueError):
            snl.estimate(model=SIR, tau=1440)
        snl.estimate(SIR, timeout=1, timeout_iteration=1)
        # Estimation history
        snl.estimate_history(phase="last")
        # Estimation accuracy
        snl.estimate_accuracy(phase="last")
        # Get a value
        snl.get(Term.RT)
        with pytest.raises(KeyError):
            snl.get("feeling")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_estimate_tau(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.trend(show_figure=False)
        with pytest.raises(ValueError):
            snl.estimate(SIRF, tau=1440, timeout=20)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_simulate(self, jhu_data, population_data, country):
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01May2020"
        with pytest.raises(ValueError):
            snl.simulate()
        with pytest.raises(ValueError):
            snl.track()
        snl.trend(show_figure=False)
        # Parameter estimation
        with pytest.raises(ValueError):
            # Deprecated
            snl.param_history(["rho"])
        all_phases = snl.summary().index.tolist()
        snl.disable(all_phases[:-2])
        with pytest.raises(NameError):
            snl.simulate()
        snl.estimate(SIRF, timeout=5, timeout_iteration=5)
        # Simulation
        snl.simulate()
        # Parameter history (Deprecated)
        snl.param_history([Term.RT], divide_by_first=False)
        snl.param_history(["rho"])
        snl.param_history(["rho"], show_figure=False)
        snl.param_history(["rho"], show_box_plot=False)
        with pytest.raises(KeyError):
            snl.param_history(["feeling"])
        # Comparison of scenarios
        snl.describe()
        snl.track()
        snl.history(target="Rt")
        snl.history(target="sigma")
        snl.history(target="rho", show_figure=False)
        snl.history(target="Infected")
        with pytest.raises(KeyError):
            snl.history(target="temperature")
        # Change rate of parameters
        snl.history_rate(name="Main")
        snl.history_rate(
            name="Main", params=["theta", "kappa"], show_figure=False)
        with pytest.raises(TypeError):
            snl.history_rate(params="", name="Main")
        # Add new scenario
        snl.add(end_date="01Sep2020", name="New")
        snl.describe()

    @pytest.mark.parametrize("country", ["Japan"])
    def test_retrospective(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Jun2020"
        snl.trend(show_figure=False)
        # Retrospective analysis
        snl.retrospective(
            "01May2020", model=SIRF, control="Main", target="Retrospective",
            timeout=1, timeout_iteration=1)

    @pytest.mark.parametrize("country", ["Italy"])
    def test_score(self, jhu_data, population_data, country):
        snl = Scenario(jhu_data, population_data, country, tau=360)
        snl.trend(show_figure=False)
        snl.estimate(SIRF, timeout=1, timeout_iteration=1)
        assert isinstance(snl.score(metrics="RMSLE"), float)
        # Selected phases
        df = snl.summary()
        all_phases = df.index.tolist()
        sel_score = snl.score(phases=all_phases[-2:])
        # Selected past days (when the begging date is a start date)
        beginning_date = df.loc[df.index[-2], Term.START]
        past_days = Term.steps(beginning_date, snl.last_date, tau=1440)
        assert snl.score(past_days=past_days) == sel_score
        # Selected past days
        snl.score(past_days=60)
        with pytest.raises(ValueError):
            snl.score(phases=["1st"], past_days=60)
