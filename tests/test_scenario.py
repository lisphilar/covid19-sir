#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import Scenario
from covsirphy import Term, PopulationData, SIRF, SIRFV, ExampleData


class TestScenario(object):
    @pytest.mark.parametrize("country", ["Italy", "Japan"])
    @pytest.mark.parametrize("province", [None, "Tokyo"])
    @pytest.mark.parametrize("tau", [None, 720, 1000])
    def test_start(self, jhu_data, population_data, country, province, tau):
        if tau == 1000 or (country == "Italy" and province == "Tokyo"):
            with pytest.raises(ValueError):
                Scenario(
                    jhu_data, population_data, country, province=province, tau=tau)
            return
        Scenario(
            jhu_data, population_data, country, province=province, tau=tau)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_start_record_range(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        # Test
        snl.first_date = "01Apr2020"
        assert snl.first_date == "01APr2020"
        snl.last_date = "01May2020"
        assert snl.last_date == "01May2020"
        with pytest.raises(ValueError):
            snl.first_date = "01Mar2020"
        with pytest.raises(ValueError):
            snl.last_date = "01Jun2020"

    @pytest.mark.parametrize("country", ["Japan"])
    def test_records(self, jhu_data, population_data, country):
        warnings.filterwarnings("ignore", category=UserWarning)
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        # Test
        df = snl.records(show_figure=False)
        assert isinstance(df, pd.DataFrame)
        assert df.columns == Term.NLOC_COLUMNS
        dates = df[Term.DATE]
        assert dates.min() == Term.date_obj(snl.first_date)
        assert dates.max() == Term.date_obj(snl.last_date)
        df2 = snl.record(show_figure=True)
        assert isinstance(df2, pd.DataFrame)
        assert df2.columns == Term.NLOC_COLUMNS

    @pytest.mark.parametrize("country", ["Japan"])
    def test_add(self, jhu_data, population_data, country):
        # Setting
        snl = Scenario(jhu_data, population_data, country)
        snl.first_date = "01Apr2020"
        snl.last_date = "01Aug2020"
        # Test
        assert snl.summary().empty
        snl.add(end_date="05May2020")
        snl.add(days=20)

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
