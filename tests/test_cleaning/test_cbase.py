#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import warnings
from covsirphy import CleaningBase, SubsetNotFoundError


class TestCleaningBase(object):
    def test_cbase(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        cbase = CleaningBase(filename=None)
        with pytest.raises(SubsetNotFoundError):
            cbase.iso3_to_country("JPN")
        with pytest.raises(NotImplementedError):
            cbase.total()
        cbase.citation = "citation"
        assert cbase.citation == "citation"
