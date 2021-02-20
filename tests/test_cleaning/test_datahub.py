#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
from covsirphy import COVID19DataHub


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
