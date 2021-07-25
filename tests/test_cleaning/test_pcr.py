#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
import pandas as pd
from covsirphy import SubsetNotFoundError
from covsirphy import PCRData, Term


class TestPCRData(object):
    def test_cleaning(self, pcr_data):
        df = pcr_data.cleaned()
        assert set([Term.DATE, Term.COUNTRY, Term.PROVINCE]).issubset(df.columns)

    @pytest.mark.parametrize("country", ["Japan"])
    def test_subset(self, pcr_data, country):
        with pytest.raises(SubsetNotFoundError):
            pcr_data.subset(country, end_date="01Jan2000")
        pcr_data.subset(country)
        df = pcr_data.subset(country, end_date="01Jan2021")
        assert set([Term.DATE, Term.TESTS, Term.C, PCRData.T_DIFF]).issubset(df.columns)

    @pytest.mark.parametrize("country", ["Greece"])
    def test_subset_complement(self, pcr_data, country):
        with pytest.raises(NotImplementedError):
            pcr_data.subset_complement(country)

    @pytest.mark.parametrize("country", ["Greece"])
    def test_records(self, pcr_data, country):
        with pytest.raises(SubsetNotFoundError):
            pcr_data.records(country, end_date="01Jan2000")
        df, _ = pcr_data.records(country)
        assert set([Term.DATE, Term.TESTS, Term.C, PCRData.T_DIFF]).issubset(df.columns)

    @pytest.mark.parametrize("country", ["Greece", "Italy", "Sweden"])
    @pytest.mark.parametrize("last_date", ["21Apr2021", None])
    def test_positive_rate(self, pcr_data, country, last_date):
        warnings.simplefilter("ignore", category=UserWarning)
        pcr_data.positive_rate(country, last_date=last_date, show_figure=True)
        df = pcr_data.positive_rate(country, last_date=last_date, show_figure=False)
        assert set([PCRData.T_DIFF, PCRData.C_DIFF, PCRData.PCR_RATE]).issubset(df.columns)
        if last_date is not None:
            assert df[pcr_data.DATE].max() <= pd.to_datetime(last_date)

    def test_map(self, pcr_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        pcr_data.map(country=None)
        pcr_data.map(country="Japan")
        with pytest.raises(NotImplementedError):
            pcr_data.map(variable="Feeling")
