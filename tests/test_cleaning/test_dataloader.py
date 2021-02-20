#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataLoader, LinelistData, JHUData, CountryData, PopulationData
from covsirphy import OxCGRTData, PCRData, VaccineData


class TestDataLoader(object):
    def test_start(self):
        with pytest.raises(TypeError):
            DataLoader(directory=0)

    def test_dataloader(self, jhu_data, population_data, oxcgrt_data,
                        japan_data, linelist_data, pcr_data, vaccine_data):
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
        # Local file
        data_loader.jhu(local_file="input/covid19dh.csv")
        data_loader.population(local_file="input/covid19dh.csv")
        data_loader.oxcgrt(local_file="input/covid19dh.csv")
        data_loader.pcr(local_file="input/covid19dh.csv")
