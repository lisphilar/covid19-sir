#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import warnings
import pandas as pd
from covsirphy import CleaningBase, SIRF
from covsirphy import COVID19DataHub, DataLoader, LinelistData, ExampleData
from covsirphy import Term, JHUData, CountryData, PopulationData, OxCGRTData
from covsirphy import Word, Population


class TestCOVID19DataHub(object):
    def test_covid19dh(self):
        with pytest.raises(TypeError):
            COVID19DataHub(filename=None)
        data_hub = COVID19DataHub(
            filename=Path("input").joinpath("covid19dh.csv"))
        # Citation (with downloading), disabled to avoid downloading many times
        # assert isinstance(data_hub.primary, str)
        # Retrieve the dataset from the server
        data_hub.load(name="jhu", force=False, verbose=False)
        with pytest.raises(KeyError):
            data_hub.load(name="unknown")
        # Citation (without downloading)
        assert isinstance(data_hub.primary, str)


class TestLinelistData(object):
    def test_linelist(self):
        linelist_data = LinelistData(
            filename=Path("input").joinpath("linelist.csv"))
        with pytest.raises(NotImplementedError):
            linelist_data.total()
        assert isinstance(linelist_data.cleaned(), pd.DataFrame)
        assert isinstance(linelist_data.citation, str)


class TestDataLoader(object):
    def test_dataloader(self):
        # Create DataLoader instance
        with pytest.raises(TypeError):
            DataLoader(directory=0)
        data_loader = DataLoader(directory="input", update_interval=12)
        # List of primary sources of COVID-19 Data Hub
        assert data_loader.covid19dh_citation
        # Data loading
        assert isinstance(data_loader.jhu(), JHUData)
        assert isinstance(data_loader.population(), PopulationData)
        assert isinstance(data_loader.oxcgrt(), OxCGRTData)
        assert isinstance(data_loader.japan(), CountryData)
        assert isinstance(data_loader.linelist(), LinelistData)
        # Local file
        data_loader.jhu(local_file="input/covid19dh.csv")
        data_loader.population(local_file="input/covid19dh.csv")
        data_loader.oxcgrt(local_file="input/covid19dh.csv")


class TestObsoleted(object):
    def test_obsoleted(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        Population(filename=None)
        Word()


class TestCleaningBase(object):
    def test_cbase(self):
        cbase = CleaningBase(filename=None)
        with pytest.raises(KeyError):
            cbase.iso3_to_country("JPN")
        with pytest.raises(NotImplementedError):
            cbase.total()


class TestExampleData(object):
    def test_iso3(self):
        example_data = ExampleData()
        example_data.add(SIRF, country="Japan")
        assert example_data.country_to_iso3("Japan") == "JPN"
        example_data.add(SIRF, country="Moon")
        assert example_data.country_to_iso3("Moon") == "---"


class TestJHUData(object):
    def test_cleaning(self, jhu_data):
        assert isinstance(jhu_data.raw, pd.DataFrame)
        with pytest.raises(ValueError):
            jhu_data.cleaned(population=None)
        df = jhu_data.cleaned()
        assert set(df.columns) == set(Term.COLUMNS)
        assert isinstance(JHUData.from_dataframe(df), JHUData)

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
        df = jhu_data.to_sr("Japan", population=126_500_000)
        assert set(df.columns) == set([Term.R, Term.S])

    def test_total(self, jhu_data):
        df = jhu_data.total()
        column_set = set(Term.VALUE_COLUMNS) | set(Term.RATE_COLUMNS)
        assert set(df.columns) == column_set

    def test_countries(self, jhu_data):
        assert isinstance(jhu_data.countries(complement=False), list)
        assert isinstance(jhu_data.countries(complement=True), list)

    @pytest.mark.parametrize("country", ["UK"])
    def test_subset_complement_non_monotonic(self, jhu_data, country):
        df, is_complemented = jhu_data.subset_complement(country=country)
        assert is_complemented
        assert df[Term.C].is_monotonic_increasing

    @pytest.mark.parametrize("country", ["Netherlands", "China"])
    def test_subset_complement_full(self, jhu_data, country):
        with pytest.raises(ValueError):
            jhu_data.subset(country=country)
        df, is_complemented = jhu_data.subset_complement(country=country)
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


class TestPopulationData(object):
    def test_cleaning(self, population_data):
        df = population_data.cleaned()
        column_set = set(Term.STR_COLUMNS) | set([Term.N, Term.ISO3])
        assert set(df.columns) == column_set

    def test_total(self, population_data):
        assert isinstance(population_data.total(), int)

    def test_to_dict(self, population_data):
        assert isinstance(population_data.to_dict(country_level=True), dict)
        assert isinstance(population_data.to_dict(country_level=False), dict)

    def test_value(self, population_data):
        assert isinstance(population_data.value("Japan"), int)
        old_value = population_data.value("Japan", date="01Mar2020")
        assert isinstance(old_value, int)
        with pytest.raises(KeyError):
            population_data.value("Japan", "01Jan1000")
        population_data.value("UK")

    def test_update(self):
        population_data = PopulationData(filename=None)
        population_data.update(1000, "Moon")
        assert population_data.value("Moon") == 1000
        population_data.update(2000, "Moon")
        assert population_data.value("Moon") == 2000

    def test_countries(self, population_data):
        assert isinstance(population_data.countries(), list)


class TestOxCGRTData(object):
    def test_cleaning(self, oxcgrt_data):
        df = oxcgrt_data.cleaned()
        assert set(df.columns) == set(OxCGRTData.OXCGRT_COLS)

    def test_subset(self, oxcgrt_data):
        with pytest.raises(KeyError):
            oxcgrt_data.subset("Moon")
        df = oxcgrt_data.subset("JPN")
        assert set(df.columns) == set(OxCGRTData.OXCGRT_COLS_WITHOUT_COUNTRY)

    def test_total(self, oxcgrt_data):
        with pytest.raises(NotImplementedError):
            oxcgrt_data.total()


class TestCountryData(object):
    def test_cleaning(self, japan_data):
        assert isinstance(japan_data.raw_columns(), list)
        df = japan_data.cleaned()
        assert set(df.columns) == set(Term.COLUMNS)

    def test_total(self, japan_data):
        df = japan_data.total()
        column_set = set(Term.VALUE_COLUMNS) | set(Term.RATE_COLUMNS)
        assert set(df.columns) == column_set

    def test_countries(self, japan_data):
        assert [japan_data.country] == japan_data.countries()

    def test_create(self):
        country_data = CountryData(filename=None, country="Moon")
        with pytest.raises(ValueError):
            country_data.cleaned()
