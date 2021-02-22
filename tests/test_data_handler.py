#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataHandler, JHUData, PopulationData
from covsirphy import UnExpectedValueError


class TestDataHandler(object):
    @pytest.mark.parametrize("country", ["Japan"])
    def test_register(self, data, country):
        dhl = DataHandler(country=country, province=None)
        if isinstance(data, JHUData):
            return dhl.register(jhu_data=data)
        if isinstance(data, PopulationData):
            return dhl.register(population_data=data)
        if type(data) in DataHandler.EXTRA_DICT.values():
            return dhl.register(extras=[data])
        with pytest.raises(UnExpectedValueError):
            dhl.register(extras=[data])
