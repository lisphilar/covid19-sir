#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import warnings
import pytest
from covsirphy import ScenarioNotFoundError, UnExecutedError, NotInteractiveError
from covsirphy import Scenario, DataHandler
from covsirphy import Term, SIRF


@pytest.fixture(scope="module")
def snl(jhu_data, population_data):
    return Scenario(
        jhu_data=jhu_data, population_data=population_data,
        country="Japan", province=None, tau=None, auto_complement=True)


class TestDataHandler(object):
    @pytest.mark.parametrize("country", ["Italy"])
    @pytest.mark.parametrize("province", [None, "Abruzzo"])
    def test_start(self, jhu_data, population_data, country, province):
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
    def test_interactive(self, jhu_data, population_data, country):
        # Setting
        dhl = DataHandler(jhu_data, population_data, country)
        dhl.init_records()
        # Force interactive
        dhl._interactive = True
        warnings.filterwarnings("ignore", category=UserWarning)
        dhl.records(show_figure=True)


class TestScenario(object):
    @pytest.mark.parametrize("start_date", ["01Mar2020"])
    @pytest.mark.parametrize("end_date", ["31Dec2020"])
    def test_record_range(self, snl, start_date, end_date):
        snl.first_date = start_date
        snl.last_date = end_date
        assert snl.first_date == start_date
        assert snl.last_date == end_date
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

    def test_complement_reverse(self, snl):
        snl.complement_reverse()
        snl.records()
        snl.records_diff()

    def test_complement(self, snl):
        snl.complement()
        snl.show_complement()
        snl.records()
        snl.records_diff()
