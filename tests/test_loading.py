#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import DBLockedError, NotDBLockedError, UnExpectedValueError
from covsirphy import DataLoader
from covsirphy import JHUData, JapanData
from covsirphy import OxCGRTData, PCRData, VaccineData, MobilityData, PopulationPyramidData
from covsirphy import Scenario, Term


class TestDataLoader(object):
    def test_start(self):
        with pytest.raises(TypeError):
            DataLoader(directory=0)

    @pytest.mark.parametrize("country", ["JPN", None])
    def test_remote(self, data_loader, country):
        loader = DataLoader(directory="input", country=country)
        # Data loading
        assert isinstance(loader.jhu(), JHUData)
        assert isinstance(loader.oxcgrt(), OxCGRTData)
        assert isinstance(loader.japan(), JapanData)
        assert isinstance(loader.pcr(), PCRData)
        assert isinstance(loader.vaccine(), VaccineData)
        assert isinstance(loader.mobility(), MobilityData)
        assert isinstance(loader.pyramid(), PopulationPyramidData)
        # List of primary sources of COVID-19 Data Hub
        assert isinstance(data_loader.covid19dh_citation, str)
        # Use all datasets in scenario analysis
        data_dict = data_loader.collect()
        snl = Scenario(country="Japan")
        snl.register(**data_dict)

    def test_local(self):
        loader = DataLoader(directory="input", update_interval=None)
        # Read CSV file: Japan dataset at country level
        loader.read_csv(JapanData.URL_C, dayfirst=False)
        loader.read_dataframe(loader.local, how_combine="replace")
        # Read CSV file: Japan dataset at province level
        loader.read_csv(JapanData.URL_P, how_combine="concat", dayfirst=False)
        with pytest.raises(UnExpectedValueError):
            loader.read_csv(JapanData.URL_P, how_combine="very-specific-way", dayfirst=False)
        # Assign country/area column
        loader.assign(country="Japan")
        loader.assign(area=lambda x: x["Location"].fillna(x["Prefecture"]))
        # Check local database (before local database lock)
        before_df = loader.local.copy()
        assert {"Date", "country", "area"}.issubset(before_df.columns)
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
        assert {Term.DATE, Term.COUNTRY, Term.PROVINCE}.issubset(locked_df.columns)
        # Create datasets
        loader.japan()
        loader.pyramid()
