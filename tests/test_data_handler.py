#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import SubsetNotFoundError
import pytest
from covsirphy import DataHandler, JHUData, PopulationData, Term
from covsirphy import CountryData, JapanData, OxCGRTData, PCRData, VaccineData
from covsirphy import UnExpectedValueError, UnExecutedError


class TestDataHandler(object):
    @pytest.mark.parametrize("country", ["Japan"])
    def test_register(self, data, country):
        dhl = DataHandler(country=country, province=None)
        # Main datasets
        if isinstance(data, JHUData):
            return dhl.register(jhu_data=data)
        if isinstance(data, PopulationData):
            return dhl.register(population_data=data)
        # Extra datasets
        if type(data) in [CountryData, JapanData, OxCGRTData, PCRData, VaccineData]:
            return dhl.register(extras=[data])
        # Un-acceptable datasets
        with pytest.raises(UnExpectedValueError):
            dhl.register(extras=[data])

    @pytest.mark.parametrize("country", ["Moon"])
    def test_register_unknown_area(self, jhu_data, population_data, country):
        dhl = DataHandler(country=country, province=None)
        dhl.register(jhu_data=jhu_data)
        with pytest.raises(SubsetNotFoundError):
            dhl.register(population_data=population_data)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_records_main(self, jhu_data, population_data, country):
        with pytest.raises(UnExecutedError):
            dhl_error = DataHandler(country=country, province=None)
            dhl_error.records_main()
        dhl = DataHandler(
            country=country, province=None, jhu_data=jhu_data, population_data=population_data)
        dhl.records_main()

    @pytest.mark.parametrize("country", ["Japan"])
    def test_complement(self, jhu_data, population_data, country):
        dhl = DataHandler(country=country, province=None)
        with pytest.raises(UnExecutedError):
            assert dhl.complemented is None
        dhl.register(jhu_data=jhu_data, population_data=population_data)
        dhl.switch_complement(whether=False)
        dhl.records_main()
        assert not dhl.complemented
        dhl.switch_complement(whether=True)
        dhl.records_main()
        assert dhl.complemented

    @pytest.mark.parametrize("country", ["Japan"])
    def test_timepoints(self, jhu_data, population_data, country):
        dhl = DataHandler(
            country=country, province=None, jhu_data=jhu_data, population_data=population_data)
        dhl.timepoints(first_date="01Apr2020", last_date="01Sep2020", today="01Jun2020")
        series = dhl.records_main()[Term.DATE]
        assert series.min().strftime(Term.DATE_FORMAT) == dhl.first_date == "01Apr2020"
        assert series.max().strftime(Term.DATE_FORMAT) == dhl.last_date == "01Sep2020"
        assert dhl.today == "01Jun2020"

    @pytest.mark.parametrize("country", ["Japan", "France"])
    def test_records_extra(self, jhu_data, population_data, country,
                           japan_data, oxcgrt_data, pcr_data, vaccine_data):
        dhl = DataHandler(
            country=country, province=None, jhu_data=jhu_data, population_data=population_data)
        dhl.timepoints(first_date="01Apr2020", last_date="01Sep2020")
        with pytest.raises(UnExecutedError):
            dhl.records_extra()
        dhl.register(extras=[japan_data, oxcgrt_data, pcr_data, vaccine_data])
        series = dhl.records_extra()[Term.DATE]
        assert series.min().strftime(Term.DATE_FORMAT) == dhl.first_date == "01Apr2020"
        assert series.max().strftime(Term.DATE_FORMAT) == dhl.last_date == "01Sep2020"
