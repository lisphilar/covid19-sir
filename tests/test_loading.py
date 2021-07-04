#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import DBLockedError, NotDBLockedError, UnExpectedValueError
from covsirphy import DataLoader
from covsirphy import JHUData, CountryData, PopulationData, JapanData
from covsirphy import OxCGRTData, PCRData, VaccineData, PopulationPyramidData
from covsirphy import Scenario, Term


class TestDataLoader(object):
    def test_start(self):
        with pytest.raises(TypeError):
            DataLoader(directory=0)

    def test_remote(self, data_loader, jhu_data, population_data, oxcgrt_data,
                    japan_data, pcr_data, vaccine_data, pyramid_data):
        # Data loading
        assert isinstance(jhu_data, JHUData)
        assert isinstance(population_data, PopulationData)
        assert isinstance(oxcgrt_data, OxCGRTData)
        assert isinstance(japan_data, CountryData)
        assert isinstance(pcr_data, PCRData)
        assert isinstance(vaccine_data, VaccineData)
        assert isinstance(pyramid_data, PopulationPyramidData)
        # List of primary sources of COVID-19 Data Hub
        assert isinstance(data_loader.covid19dh_citation, str)
        # Collect datasets for scenario analysis
        data_dict = data_loader.collect()
        snl = Scenario(country="Japan")
        snl.register(**data_dict)

    def test_local_and_remote(self):
        loader = DataLoader(directory="input")
        # Read CSV file: Japan dataset at country level
        loader.read_csv(JapanData.URL_C, dayfirst=False)
        # Read CSV file: Japan dataset at province level
        loader.read_csv(JapanData.URL_P, how_combine="concat", dayfirst=False)
        with pytest.raises(UnExpectedValueError):
            loader.read_csv(JapanData.URL_P, how_combine="very-specific-way", dayfirst=False)
        # Assign country/arwa column
        loader.assign(country="Japan")
        loader.assign(area=lambda x: x["Location"].fillna(x["Prefecture"]))
        # Check local database (before local database lock)
        before_df = loader.local.copy()
        assert set(["Date", "country", "area"]).issubset(before_df.columns)
        # Local database lock
        with pytest.raises(NotDBLockedError):
            assert isinstance(loader.locked, pd.DataFrame)
        loader.lock(
            date="Date", country="country", province="area",
            confirmed="Positive", tests="Tested", recovered="Discharged")
        with pytest.raises(DBLockedError):
            loader.lock(date="Date", country="country", province="area")
        # Check locked database (after database lock)
        locked_df = loader.locked.copy()
        assert set([Term.DATE, Term.COUNTRY, Term.PROVINCE]).issubset(locked_df.columns)
        # Create datasets
        loader.japan()
        loader.pyramid()
        data_dict = loader.collect()
        snl = Scenario(country="Italy")
        snl.register(**data_dict)
