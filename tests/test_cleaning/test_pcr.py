#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
from covsirphy import SubsetNotFoundError, PCRIncorrectPreconditionError
from covsirphy import PCRData


class TestPCRData(object):
    def test_cleaning(self, pcr_data):
        df = pcr_data.cleaned()
        assert set(df.columns) == set(PCRData.PCR_COLUMNS)

    def test_from_dataframe(self, pcr_data):
        df = pcr_data.cleaned()
        assert isinstance(PCRData.from_dataframe(df), PCRData)

    def test_use_ourworldindata(self, pcr_data):
        pcr_data.use_ourworldindata(
            filename="input/ourworldindata_pcr.csv")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_subset(self, pcr_data, country):
        with pytest.raises(SubsetNotFoundError):
            pcr_data.subset(country, end_date="01Jan2000")
        df = pcr_data.subset(country)
        df = pcr_data.subset(country, end_date="01Jan2021")
        assert set(df.columns) == set([*PCRData.PCR_NLOC_COLUMNS, PCRData.T_DIFF])

    @pytest.mark.parametrize("country", ["Greece"])
    def test_subset_complement(self, pcr_data, country):
        with pytest.raises(NotImplementedError):
            pcr_data.subset_complement(country)

    @pytest.mark.parametrize("country", ["Greece"])
    def test_records(self, pcr_data, country):
        with pytest.raises(SubsetNotFoundError):
            pcr_data.records(country, end_date="01Jan2000")
        df, _ = pcr_data.records(country)
        assert set(df.columns) == set([*PCRData.PCR_NLOC_COLUMNS, PCRData.T_DIFF])

    @pytest.mark.parametrize("country", ["Greece", "Italy", "Sweden"])
    def test_positive_rate(self, pcr_data, country):
        warnings.simplefilter("ignore", category=UserWarning)
        pcr_data.positive_rate(country, show_figure=True)
        df = pcr_data.positive_rate(country, show_figure=False)
        assert set([PCRData.T_DIFF, PCRData.C_DIFF, PCRData.PCR_RATE]).issubset(df.columns)

    @pytest.mark.parametrize("country", ["China"])
    def test_positive_rate_error(self, pcr_data, country):
        with pytest.raises(PCRIncorrectPreconditionError):
            pcr_data.positive_rate(country, show_figure=False)

    def test_map(self, pcr_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        pcr_data.map(country=None)
        pcr_data.map(country="Japan")
        with pytest.raises(NotImplementedError):
            pcr_data.map(variable="Feeling")
