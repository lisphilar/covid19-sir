#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import PolicyMeasures
from covsirphy import SIRF, Scenario


class TestPolicyMeasures(object):
    def test_start(self, jhu_data, population_data, oxcgrt_data):
        warnings.simplefilter("ignore", category=UserWarning)
        # Create instance
        analyser = PolicyMeasures(
            jhu_data, population_data, oxcgrt_data, tau=360)
        # List of countries
        assert isinstance(analyser.countries, list)
        # Return Scenario class
        assert isinstance(analyser.scenario("Japan"), Scenario)
        with pytest.raises(KeyError):
            analyser.scenario("Moon")
        assert isinstance(analyser.countries, list)
        # S-R trend analysis
        analyser.trend()
        min_len = max(analyser.phase_len().keys())
        analyser.trend(min_len=min_len)
        # Summarize
        assert isinstance(analyser.summary(), pd.DataFrame)
        with pytest.raises(TypeError):
            analyser.summary(countries="Poland")
        # Phase length
        phase_len_dict = analyser.phase_len()
        assert isinstance(phase_len_dict, dict)
        assert isinstance(phase_len_dict[min_len], list)

    def test_analysis(self, jhu_data, population_data, oxcgrt_data):
        warnings.simplefilter("ignore", category=UserWarning)
        # Create instance
        analyser = PolicyMeasures(
            jhu_data, population_data, oxcgrt_data, tau=360)
        # S-R trend analysis
        analyser.trend()
        # Select two countries
        phase_len_dict = analyser.phase_len()
        countries_all = [
            country
            for (num, countries) in sorted(phase_len_dict.items(), reverse=True)
            for country in countries
        ]
        analyser.countries = countries_all[:2]
        # Parameter estimation
        with pytest.raises(ValueError):
            analyser.track()
        analyser.estimate(SIRF, timeout=10)
        assert isinstance(analyser.summary(), pd.DataFrame)
        # Parameter history of Rt
        with pytest.raises(KeyError):
            df = analyser.history("Temperature", roll_window=None)
        df = analyser.history("Rt", roll_window=None)
        assert isinstance(df, pd.DataFrame)
        # Parameter history of rho
        df = analyser.history("rho", roll_window=14, show_figure=False)
        assert isinstance(df, pd.DataFrame)

    def test_error(self, jhu_data, population_data, oxcgrt_data):
        warnings.simplefilter("ignore", category=UserWarning)
        # Create instance
        analyser = PolicyMeasures(
            jhu_data, population_data, oxcgrt_data, tau=360)
        # Register countries
        with pytest.raises(KeyError):
            analyser.countries = ["Moon"]
