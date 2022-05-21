#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import save_dataframe, Filer


class TestFiler(object):
    def test_filing(self):
        with pytest.raises(ValueError):
            Filer("output", numbering="xxx")
        filer = Filer(directory=["output", "A"], prefix="jpn", suffix=None, numbering="01")
        filer = Filer(directory=("output", "A", "B"), prefix="jpn", suffix=None, numbering="01")
        filer = Filer(directory="output", prefix="jpn", suffix=None, numbering="01")
        # Create filenames
        filer.png("records")
        filer.jpg("records")
        filer.json("records")
        filer.geojson("records")
        filer.csv("records", index=True)
        # Check files
        assert len(filer.files(ext=None)) == 5
        assert len(filer.files(ext="png")) == 1
        assert len(filer.files(ext="jpg")) == 1
        assert len(filer.files(ext="json")) == 1
        assert len(filer.files(ext="geojson")) == 1
        assert len(filer.files(ext="csv")) == 1
        # Save CSV file
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        save_dataframe(pd.DataFrame(), filename=None, index=False)
