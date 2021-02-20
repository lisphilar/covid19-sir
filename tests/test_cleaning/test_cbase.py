#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import CleaningBase, Word, Population


class TestCleaningBase(object):
    def test_cbase(self):
        cbase = CleaningBase(filename=None)
        with pytest.raises(KeyError):
            cbase.iso3_to_country("JPN")
        with pytest.raises(NotImplementedError):
            cbase.total()
        cbase.citation = "citation"
        assert cbase.citation == "citation"


class TestObsoleted(object):
    def test_obsoleted(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        Population(filename=None)
        Word()
