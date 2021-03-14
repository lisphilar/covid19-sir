#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pytest
import pandas as pd
from covsirphy import SubsetNotFoundError, Term, JHUData
from covsirphy import JHUDataComplementHandler


class TestJHUData(object):
    def test_cleaning(self, jhu_data):
        assert isinstance(jhu_data.raw, pd.DataFrame)
        with pytest.raises(ValueError):
            jhu_data.cleaned(population=None)
        df = jhu_data.cleaned()
        assert set(df.columns) == set(Term.COLUMNS)
        assert isinstance(JHUData.from_dataframe(df), JHUData)

    def test_from_dataframe(self, japan_data):
        df = japan_data.cleaned()
        jhu_data_df = JHUData.from_dataframe(df, directory="input_dir")
        assert isinstance(jhu_data_df, JHUData)
        assert jhu_data_df.directory == "input_dir"
        jhu_data_df.records("Japan")

    def test_subset(self, jhu_data):
        df = jhu_data.subset(
            "Japan", province="Tokyo", start_date="01Apr2020", end_date="01Jun2020")
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
        with pytest.raises(KeyError):
            jhu_data.subset("Moon")
        with pytest.raises(ValueError):
            jhu_data.subset(
                "Japan", start_date="01Jan2020", end_date="10Jan2020")
        s_df = jhu_data.subset("Japan", population=126_500_000)
        assert set(s_df.columns) == set(Term.SUB_COLUMNS)
        jhu_data.subset("US")

    def test_replace(self, jhu_data, japan_data):
        jhu_data.replace(japan_data)
        df = jhu_data.subset("Japan")
        japan_df = japan_data.cleaned()
        last_date = japan_df.loc[japan_df.index[-1], Term.DATE]
        assert df.loc[df.index[-1], Term.DATE] == last_date

    def test_to_sr(self, jhu_data):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        df = jhu_data.to_sr("Japan", population=126_500_000)
        assert set(df.columns) == set([Term.R, Term.S])

    def test_total(self, jhu_data):
        df = jhu_data.total()
        column_set = set(Term.VALUE_COLUMNS) | set(Term.RATE_COLUMNS)
        assert set(df.columns) == column_set

    def test_countries(self, jhu_data):
        assert isinstance(jhu_data.countries(complement=False), list)
        assert isinstance(jhu_data.countries(complement=True), list)

    def test_closing_period(self, jhu_data):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        assert isinstance(jhu_data.calculate_closing_period(), int)
        assert isinstance(jhu_data.calculate_recovery_period(), int)

    @pytest.mark.parametrize("country", ["UK"])
    def test_subset_complement_non_monotonic(self, jhu_data, country):
        df, is_complemented = jhu_data.subset_complement(country=country)
        assert is_complemented
        assert df[Term.C].is_monotonic_increasing

    @pytest.mark.parametrize("country", ["Netherlands", "Germany"])
    def test_subset_complement_full(self, jhu_data, country):
        if country in set(["Netherlands"]):
            with pytest.raises(ValueError):
                jhu_data.subset(country=country)
        df, is_complemented = jhu_data.subset_complement(country=country)
        assert isinstance(jhu_data.recovery_period, int)
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
        assert is_complemented
        with pytest.raises(KeyError):
            jhu_data.subset_complement(country=country, end_date="01Jan1900")

    @pytest.mark.parametrize("country", ["Japan"])
    def test_subset_complement_partial(self, jhu_data, country):
        df, is_complemented = jhu_data.subset_complement(country=country)
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
        assert is_complemented
        with pytest.raises(KeyError):
            jhu_data.subset_complement(country=country, end_date="01Jan1900")

    @pytest.mark.parametrize(
        "country", ["UK", "Netherlands", "China", "Germany", "France", "Japan"])
    def test_records(self, jhu_data, country):
        df, is_complemented = jhu_data.records(country=country)
        assert set(df.columns) == set(Term.NLOC_COLUMNS)
        assert is_complemented

    @pytest.mark.parametrize("country", ["Netherlands", "Moon"])
    def test_records_error(self, jhu_data, country):
        with pytest.raises(SubsetNotFoundError):
            jhu_data.records(country=country, auto_complement=False)

    @pytest.mark.parametrize(
        "applied, expected, iso3",
        [
            ("Congo", "Republic of the Congo", "COG"),
            ("Democratic Congo", "Democratic Republic of the Congo", "COD"),
            ("GR", "Greece", "GRC"),
            ("gr", "Greece", "GRC"),
            ("GRC", "Greece", "GRC"),
            ("Greece", "Greece", "GRC"),
            ("GREECE", "Greece", "GRC"),
            ("gre", "error", "GRC"),
            ("Ivory Coast", "Cote d'Ivoire", "CIV"),
            ("Korea, South", "South Korea", "KOR"),
            ("UK", "United Kingdom", "GBR"),
            ("US", "United States", "USA"),
            ("USA", "United States", "USA"),
            ("VAT", "Holy See", "VAT"),
        ]
    )
    def test_country_name(self, jhu_data, applied, expected, iso3):
        if expected == "error":
            with pytest.raises(KeyError):
                jhu_data.ensure_country_name(applied)
        else:
            response = jhu_data.ensure_country_name(applied)
            assert response == expected
            assert jhu_data.country_to_iso3(response) == iso3

    @pytest.mark.parametrize(
        "country, province",
        [
            ("Greece", None),
            (["Greece", "Japan"], None),
            (None, None),
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
