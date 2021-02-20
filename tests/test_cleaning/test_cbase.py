#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import CleaningBase, Word, Population, SubsetNotFoundError


class TestCleaningBase(object):
    def test_cbase(self):
        cbase = CleaningBase(filename=None)
        with pytest.raises(KeyError):
            cbase.iso3_to_country("JPN")
        with pytest.raises(NotImplementedError):
            cbase.total()
        cbase.citation = "citation"
        assert cbase.citation == "citation"

    @pytest.mark.parametrize("country", [None, "Japan"])
    def test_layer(self, data, country):
        # Country level data
        data.layer(country=None)
        # Province level data
        data.layer(country=country)

    @pytest.mark.parametrize("country", ["Moon"])
    def test_layer_error(self, japan_data, country):
        with pytest.raises(SubsetNotFoundError):
            japan_data.layer(country=country)


class TestObsoleted(object):
    def test_obsoleted(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        Population(filename=None)
        Word()
