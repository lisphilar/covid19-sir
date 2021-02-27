#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import warnings
import pytest
import pandas as pd
from covsirphy import ScenarioNotFoundError, UnExecutedError, NotInteractiveError
from covsirphy import Scenario, Term, PhaseSeries, Estimator, SIRF


@pytest.fixture(scope="module")
def snl(jhu_data, population_data, japan_data, oxcgrt_data, vaccine_data, pcr_data):
    snl = Scenario(country="Japan", province=None, tau=None, auto_complement=True)
    snl.register(
        jhu_data, population_data,
        extras=[japan_data, oxcgrt_data, pcr_data, vaccine_data])
    return snl


class TestScenario(object):
    @pytest.mark.parametrize("first_date", ["01Mar2020"])
    @pytest.mark.parametrize("last_date", ["31Dec2020"])
    @pytest.mark.parametrize("today", ["30Nov2020"])
    def test_record_range(self, snl, first_date, last_date, today):
        snl.timepoints()
        snl.first_date = first_date
        snl.last_date = last_date
        snl.today = today
        assert snl.first_date == first_date
        assert snl.last_date == last_date
        assert snl.today == today
        with pytest.raises(ValueError):
            snl.first_date = "01Jan2019"
        with pytest.raises(ValueError):
            snl.last_date = Term.tomorrow(datetime.now().strftime(Term.DATE_FORMAT))

    def test_line_plot(self, snl, imgfile):
        warnings.simplefilter("ignore", category=UserWarning)
        # Interactive / script mode
        assert not snl.interactive
        with pytest.raises(NotInteractiveError):
            snl.interactive = True
        snl.interactive = False
        # Change colors in plotting
        snl.records(
            variables=["Confirmed", "Infected", "Fatal", "Recovered"],
            color_dict={"Confirmed": "blue", "Infected": "orange", "Fatal": "red", "Recovered": "green"},
            filename=imgfile,
        )

    @pytest.mark.parametrize("country", ["Japan"])
    def test_interactive(self, jhu_data, population_data, country):
        with pytest.raises(ValueError):
            Scenario()
        # Setting
        scenario = Scenario(country=country)
        scenario.register(jhu_data, population_data)
        # Force interactive
        scenario._interactive = True
        warnings.filterwarnings("ignore", category=UserWarning)
        scenario.records(show_figure=True)

    def test_records(self, snl):
        # Not complemented
        snl.complement_reverse()
        snl.records(variables=None)
        snl.records(variables="all")
        df = snl.records(variables=[Term.TESTS, Term.VAC])
        assert set(df.columns) == set([Term.DATE, Term.TESTS, Term.VAC])
        snl.records_diff()
        # Complemented
        snl.complement()
        snl.records()
        snl.records_diff(variables=None)
        snl.records_diff(variables="all")
        diff_df = snl.records_diff(variables=[Term.TESTS, Term.VAC])
        assert set(diff_df.columns) == set([Term.TESTS, Term.VAC])
        # Details of complement
        snl.show_complement()

    def test_register_scenario(self, snl):
        with pytest.raises(ScenarioNotFoundError):
            snl.clear(name="New", include_past=True, template="Un-registered")
        snl.clear(name="New", include_past=True, template="Main")
        assert isinstance(snl["New"], PhaseSeries)
        snl.summary()
        snl.delete(name="New")
        with pytest.raises(ScenarioNotFoundError):
            snl["New"]

    def test_add_delete(self, snl):
        # Add phases
        snl.add(end_date="01Sep2020")
        with pytest.raises(ValueError):
            snl.add(end_date="01Jun2020")
        # Deprecated
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        snl.add_phase(end_date="01Oct2020")
        # Add a phase with days
        snl.add(days=10)
        # Add a phase to the last date
        snl.add()
        # Delete phases
        snl.delete(phases=["last"])
        snl.clear(include_past=False)
        snl.delete(phases=None)
        snl.clear(include_past=True)

    def test_enable(self, snl):
        # Setting
        snl.add(end_date="01May2020")
        snl.add(end_date="01Sep2020")
        snl.add(end_date="01Nov2020")
        # Disable/enable
        snl.disable(phases=["1st", "2nd"])
        snl.enable(phases=["2nd"])
        # Test (only 1st phase is disabled)
        assert "0th" in snl.summary().index
        assert "1st" not in snl.summary().index
        assert "2nd" in snl.summary().index
        # Clear all phases
        snl.clear(include_past=True)

    def test_combine_separate(self, snl):
        # Setting
        snl.add(end_date="01May2020")
        snl.add(end_date="01Sep2020")
        snl.add(end_date="01Nov2020")
        # Combine
        snl.combine(phases=["0th", "1st"])
        snl.separate(date="01Jun2020")
        assert len(snl.summary()) == 3
        # Clear all phases
        snl.clear(include_past=True)

    def test_trend(self, snl):
        snl.trend()
        with pytest.raises(ValueError):
            snl.trend(n_points=2)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        snl.trend(force=False, include_init_phase=False)

    def test_estimate(self, snl):
        # Error test
        with pytest.raises(UnExecutedError):
            snl.phase_estimator(phase="1st")
        with pytest.raises(UnExecutedError):
            snl.simulate()
        with pytest.raises(UnExecutedError):
            snl.fit()
        with pytest.raises(ValueError):
            snl.estimate(SIRF, tau=1440)
        # Deprecated
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with pytest.raises(UnExecutedError):
            snl.param_history()
        # Parameter estimation
        snl.estimate(SIRF, timeout=1, timeout_interation=1)
        snl.summary()

    def test_estimator(self, snl):
        assert isinstance(snl.phase_estimator(phase="1st"), Estimator)
        snl.estimate_history(phase="1st")
        snl.estimate_accuracy(phase="1st")

    def test_simulate(self, snl):
        snl.simulate()
        snl.simulate(phases=["1st", "2nd"])

    def test_get(self, snl):
        with pytest.raises(KeyError):
            snl.get("feeling", phase="last")
        assert isinstance(snl.get("rho", phase="last"), float)

    def test_param_history(self, snl):
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        snl.param_history()
        with pytest.raises(KeyError):
            snl.param_history(targets=["feeling"])
        snl.param_history(divide_by_first=False, show_figure=False)
        snl.param_history(show_box_plot=False)

    def test_describe(self, snl):
        snl.add(days=100)
        snl.describe()
        snl.clear(name="New", include_past=False, template="Main")
        snl.add(days=100, rho=0.01, name="New")
        snl.describe()
        snl.delete(name="New")
        snl.clear(name="Main")

    def test_track(self, snl):
        snl.track()

    @pytest.mark.parametrize("target", ["rho", "Infected", "Rt"])
    def test_history(self, snl, target):
        snl.history(target=target)

    def test_history_error(self, snl):
        with pytest.raises(KeyError):
            snl.history(target="feeling")

    def test_history_rate(self, snl):
        snl.history_rate(params=None)
        snl.history_rate(params=["rho", "sigma"])
        with pytest.raises(TypeError):
            snl.history_rate(params="rho")

    def test_retrospective(self, snl):
        date = snl.summary().loc["5th", Term.START]
        snl.clear(name="Control", template="Main")
        snl.retrospective(
            beginning_date=date, model=SIRF,
            control="Control", target="Retro", timeout=1, timeout_iteration=1)

    @pytest.mark.parametrize("metrics", ["MAE", "MSE", "MSLE", "RMSE", "RMSLE"])
    def test_score(self, snl, metrics):
        try:
            snl.delete(name="Score")
        except KeyError:
            pass
        snl.clear(name="Score", template="Main")
        assert isinstance(snl.score(metrics=metrics, name="Score"), float)
        # Selected phases
        df = snl.summary(name="Score")
        all_phases = df.index.tolist()
        sel_score = snl.score(phases=all_phases[-2:], name="Score")
        # Selected past days (when the begging date is a start date)
        beginning_date = df.loc[df.index[-2], Term.START]
        past_days = Term.steps(beginning_date, snl.last_date, tau=1440)
        assert snl.score(past_days=past_days, name="Score") == sel_score
        # Selected past days
        snl.score(past_days=60, name="Score")
        with pytest.raises(ValueError):
            snl.score(phases=["1st"], past_days=60, name="Score")

    @pytest.mark.parametrize("indicator", ["Stringency_index"])
    @pytest.mark.parametrize("target", ["Confirmed"])
    def test_estimate_delay(self, snl, indicator, target, oxcgrt_data):
        warnings.simplefilter("ignore", category=UserWarning)
        delay, df = snl.estimate_delay(indicator=indicator, target=target)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        delay, df = snl.estimate_delay(oxcgrt_data, indicator=indicator, target=target, value_range=(7, 35))
        assert isinstance(delay, int)
        assert isinstance(df, pd.DataFrame)

    def test_fit_predict(self, snl, oxcgrt_data):
        # Fitting
        with pytest.raises(UnExecutedError):
            snl.predict()
        info_dict = snl.fit()
        assert isinstance(info_dict, dict)
        # Deprecated: fit with oxcgrt_data
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        snl.fit(oxcgrt_data)
        # Prediction
        snl.predict()
        assert Term.FUTURE in snl.summary()[Term.TENSE].unique()
        # Fitting & predict
        snl.fit_predict()
        assert Term.FUTURE in snl.summary()[Term.TENSE].unique()
