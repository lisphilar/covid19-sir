#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import pytest
import warnings
from covsirphy import CleaningBase, Word, Population, SubsetNotFoundError


class TestCleaningBase(object):
    def test_cbase(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        cbase = CleaningBase(filename=None)
        with pytest.raises(KeyError):
            cbase.iso3_to_country("JPN")
        with pytest.raises(NotImplementedError):
            cbase.total()
        cbase.citation = "citation"
        assert cbase.citation == "citation"

    @pytest.mark.parametrize("geo", [None, ("Japan",)])
    def test_layer(self, data, geo):
        with contextlib.suppress(NotImplementedError):
            data.layer(geo=geo)

    @pytest.mark.parametrize("country", [None, "Japan"])
    def test_layer_deprecated(self, data, country):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        with contextlib.suppress(NotImplementedError):
            data.layer(country=country)

    @pytest.mark.parametrize("country", ["Moon"])
    def test_layer_error(self, japan_data, country):
        with pytest.raises(SubsetNotFoundError):
            japan_data.layer(country=country)


class TestObsoleted(object):
    def test_obsoleted(self):
        Population(filename=None)
        Word()
