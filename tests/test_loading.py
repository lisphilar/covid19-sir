#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import UnExpectedValueError
from covsirphy.cleaning.japan_data import JapanData
from pathlib import Path
import pytest
from covsirphy import DBLockedError, NotDBLockedError
from covsirphy import DataLoader, COVID19DataHub
from covsirphy import JHUData, CountryData, PopulationData
from covsirphy import OxCGRTData, PCRData, VaccineData, PopulationPyramidData
from covsirphy import Scenario, Term


class TestDataLoader(object):
    def test_start(self):
        with pytest.raises(TypeError):
            DataLoader(directory=0)

    def test_dataloader(self, jhu_data, population_data, oxcgrt_data,
                        japan_data, pcr_data, vaccine_data, pyramid_data):
        # List of primary sources of COVID-19 Data Hub
        data_loader = DataLoader()
        assert data_loader.covid19dh_citation
        # Data loading
        assert isinstance(jhu_data, JHUData)
        assert isinstance(population_data, PopulationData)
        assert isinstance(oxcgrt_data, OxCGRTData)
        assert isinstance(japan_data, CountryData)
        assert isinstance(pcr_data, PCRData)
        assert isinstance(vaccine_data, VaccineData)
        assert isinstance(pyramid_data, PopulationPyramidData)
        # Local file
        data_loader.jhu(local_file="input/covid19dh.csv")
        data_loader.population(local_file="input/covid19dh.csv")
        data_loader.oxcgrt(local_file="input/covid19dh.csv")
        data_loader.pcr(local_file="input/covid19dh.csv")

    def test_local(self):
        loader = DataLoader(directory="input", update_interval=None)
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
        before_df = loader.local(locked=False)
        assert set(["Date", "country", "area"]).issubset(before_df.columns)
        # Local database lock
        with pytest.raises(NotDBLockedError):
            loader.local(locked=True)
        loader.lock(
            date="Date", country="country", province="area",
            confiremd="Positive", tests="Tested", recovered="Discharged")
        with pytest.raises(DBLockedError):
            loader.lock(date="Date", country="country", province="area")
        # Check local database (after database lock)
        locked_df = loader.local(locked=True)
        assert set([Term.DATE, Term.COUNTRY, Term.PROVINCE]).issubset(locked_df.columns)

    def test_collect(self, data_loader):
        data_dict = data_loader.collect()
        snl = Scenario(country="Japan")
        snl.register(**data_dict)


@pytest.mark.filterwarnings("ignore", category=DeprecationWarning)
class TestCOVID19DataHub(object):
    def test_covid19dh(self):
        data_hub = COVID19DataHub(filename=Path("input").joinpath("covid19dh.csv"))
        data_hub.load(name="jhu")
        assert isinstance(data_hub.primary, str)
