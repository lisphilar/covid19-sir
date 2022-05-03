#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import SubsetNotFoundError, Term, JHUDataComplementHandler


class TestJHUData(object):
    def test_cleaning(self, jhu_data):
        assert isinstance(jhu_data.raw, pd.DataFrame)
        with pytest.raises(ValueError):
            jhu_data.cleaned(population=None)
        df = jhu_data.cleaned()
        assert set(df.columns) == {*Term.COLUMNS, Term.ISO3, Term.N}

    @pytest.mark.parametrize("country", ["Japan"])
    def test_subset(self, jhu_data, country):
        df = jhu_data.subset(country, start_date="01Apr2020", end_date="01Jun2020")
        assert df[Term.S].dtype == "int64"
        with pytest.raises(KeyError):
            jhu_data.subset("Moon")
        with pytest.raises(ValueError):
            jhu_data.subset(country, start_date="01Jan2020", end_date="10Jan2020")
        if country == "Japan":
            s_df = jhu_data.subset(country, population=126_500_000)
            assert int(s_df[[Term.S, Term.C]].sum(axis=1).mean()) == 126_500_000

    def test_total(self, jhu_data):
        df = jhu_data.total()
        column_set = set(Term.VALUE_COLUMNS) | set(Term.RATE_COLUMNS)
        assert set(df.columns) == column_set

    def test_countries(self, jhu_data):
        assert isinstance(jhu_data.countries(complement=False), list)

    @pytest.mark.parametrize("country", ["Netherlands", "Germany"])
    def test_subset_complement_full(self, jhu_data, country):
        if country in {"Netherlands"}:
            with pytest.raises(ValueError):
                jhu_data.subset(country=country)
        df, is_complemented = jhu_data.subset_complement(country=country)
        assert isinstance(jhu_data.recovery_period, int)
        assert is_complemented
        with pytest.raises(KeyError):
            jhu_data.subset_complement(country=country, end_date="01Jan1900")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_subset_complement_partial(self, jhu_data, country):
        df, is_complemented = jhu_data.subset_complement(country=country)
        assert is_complemented
        with pytest.raises(KeyError):
            jhu_data.subset_complement(country=country, end_date="01Jan1900")

    @pytest.mark.parametrize(
        "country", ["UK", "Netherlands", "China", "Germany", "France", "Japan"])
    def test_records(self, jhu_data, country):
        df, is_complemented = jhu_data.records(country=country)
        assert is_complemented

    @pytest.mark.parametrize("country", ["Netherlands", "Moon"])
    def test_records_error(self, jhu_data, country):
        with pytest.raises(SubsetNotFoundError):
            jhu_data.records(country=country, auto_complement=False)

    @pytest.mark.parametrize(
        "applied, iso3",
        [
            ("Congo", "COG"),
            ("Democratic Congo", "COD"),
            ("GR", "GRC"),
            ("gr", "GRC"),
            ("GRC", "GRC"),
            ("Greece", "GRC"),
            ("GREECE", "GRC"),
            ("gre", "error"),
            ("Ivory Coast", "CIV"),
            ("Korea, South", "KOR"),
            ("UK", "GBR"),
            ("US", "USA"),
            ("USA", "USA"),
            ("Holy See", "VAT"),
        ]
    )
    def test_country_name(self, jhu_data, applied, iso3):
        if iso3 == "error":
            with pytest.raises(KeyError):
                jhu_data.ensure_country_name(applied)
        else:
            assert jhu_data.ensure_country_name(applied) == iso3

    @pytest.mark.parametrize(
        "country, province",
        [
            ("Greece", None),
            (["Greece", "Japan"], None),
            # raise ValueError
            (["Greece", "Japan"], "Tokyo"),
            # raise SubsetNotFoundError
            ("Moon", None),
        ]
    )
    def test_show_complement(self, jhu_data, country, province):
        if country == "Moon":
            with pytest.raises(SubsetNotFoundError):
                jhu_data.show_complement(country=country, province=province)
        elif not isinstance(country, str) and province is not None:
            with pytest.raises(ValueError):
                jhu_data.show_complement(country=country, province=province)
        else:
            df = jhu_data.show_complement(country=country, province=province)
            all_set = set(JHUDataComplementHandler.SHOW_COMPLEMENT_FULL_COLS)
            assert all_set.issubset(df.columns)

    def test_map(self, jhu_data):
        warnings.filterwarnings("ignore", category=UserWarning)
        # Global map
        for variable in Term.VALUE_COLUMNS:
            jhu_data.map(country=None, variable=variable)
        jhu_data.map(country=None, included=["Japan", "Greece"])
        # Country map
        for variable in Term.VALUE_COLUMNS:
            jhu_data.map(country="Japan", variable=variable)
        jhu_data.map(country="Japan", excluded=["Tokyo"])
        # Error handling
        with pytest.raises(SubsetNotFoundError):
            jhu_data.map(country="Greece")
