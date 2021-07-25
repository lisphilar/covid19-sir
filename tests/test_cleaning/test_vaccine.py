#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import SubsetNotFoundError, Term


class TestVaccineData(object):
    def test_cleaning(self, vaccine_data):
        df = vaccine_data.cleaned()
        assert set([Term.DATE, Term.COUNTRY, Term.PROVINCE]).issubset(df.columns)

    @pytest.mark.parametrize("country", ["Canada"])
    def test_subset(self, vaccine_data, country):
        df = vaccine_data.subset(country=country)
        assert set(df.columns) == set([Term.DATE, Term.VAC, Term.V_ONCE, Term.V_FULL])
        clean_df = vaccine_data.cleaned()
        df = clean_df.loc[clean_df[Term.COUNTRY] == country, :]
        product = df.loc[df.index[0], Term.PRODUCT]
        vaccine_data.subset(
            country=country, product=product, start_date="01Jan2021", end_date="15Jan2021")
        with pytest.raises(SubsetNotFoundError):
            vaccine_data.subset(country=country, end_date="01May2020")

    @pytest.mark.parametrize("country", ["GBR"])
    def test_records(self, vaccine_data, country):
        vaccine_data.records(country=country)

    def test_total(self, vaccine_data):
        df = vaccine_data.total()
        assert set(df.columns) == set([Term.DATE, Term.VAC, Term.V_ONCE, Term.V_FULL])

    def test_map(self, vaccine_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        vaccine_data.map()
        with pytest.raises(NotImplementedError):
            vaccine_data.map(country="GBR")
