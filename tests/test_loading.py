#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
from covsirphy import DataLoader, COVID19DataHub
from covsirphy import LinelistData, JHUData, CountryData, PopulationData
from covsirphy import OxCGRTData, PCRData, VaccineData, PopulationPyramidData
from covsirphy import Scenario
from covsirphy.loading.loaderbase import _LoaderBase


class TestLoaderBase(object):
    def test_not_implelemted(self):
        base = _LoaderBase()
        with pytest.raises(NotImplementedError):
            base.jhu()
        with pytest.raises(NotImplementedError):
            base.population()
        with pytest.raises(NotImplementedError):
            base.oxcgrt()
        with pytest.raises(NotImplementedError):
            base.japan()
        with pytest.raises(NotImplementedError):
            base.linelist()
        with pytest.raises(NotImplementedError):
            base.pcr()
        with pytest.raises(NotImplementedError):
            base.vaccine()
        with pytest.raises(NotImplementedError):
            base.pyramid()


class TestDataLoader(object):
    def test_start(self):
        with pytest.raises(TypeError):
            DataLoader(directory=0)

    def test_dataloader(self, jhu_data, population_data, oxcgrt_data,
                        japan_data, linelist_data, pcr_data, vaccine_data, pyramid_data):
        # List of primary sources of COVID-19 Data Hub
        data_loader = DataLoader()
        assert data_loader.covid19dh_citation
        # Data loading
        assert isinstance(jhu_data, JHUData)
        assert isinstance(population_data, PopulationData)
        assert isinstance(oxcgrt_data, OxCGRTData)
        assert isinstance(japan_data, CountryData)
        assert isinstance(linelist_data, LinelistData)
        assert isinstance(pcr_data, PCRData)
        assert isinstance(vaccine_data, VaccineData)
        assert isinstance(pyramid_data, PopulationPyramidData)
        # Local file
        data_loader.jhu(local_file="input/covid19dh.csv")
        data_loader.population(local_file="input/covid19dh.csv")
        data_loader.oxcgrt(local_file="input/covid19dh.csv")
        data_loader.pcr(local_file="input/covid19dh.csv")

    def test_collect(self, data_loader):
        data_dict = data_loader.collect()
        snl = Scenario(country="Japan")
        snl.register(**data_dict)


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
