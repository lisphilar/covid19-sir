#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from covsirphy import Term, LayerAdjuster


class TestLayerAdjuster(object):
    def test_empty(self):
        with pytest.raises(ValueError):
            collector = LayerAdjuster(layers=["ISO3", "Province", "Province"])
        collector = LayerAdjuster(layers=["ISO3", "Province"])
        all_df = collector.all()
        assert all_df.empty
        assert set(all_df.columns) == {"Date", "ISO3", "Province"}
        assert not collector.citations()

    @pytest.mark.parametrize(
        "country, layers, data_dict, all_dict",
        (
            (
                "ISO3",
                ["ISO3", "Province", "City"],
                {"ISO3": ["JPN", "JPN"], "Province": ["-", "Tokyo"], "City": ["-", "Chiyoda"]},
                {"ISO3": ["JPN", "JPN"], "Province": ["-", "Tokyo"], "City": ["-", "Chiyoda"]}
            ),
            (
                "ISO3",
                ["ISO3", "Province", "City"],
                {"ISO3": ["JPN", "JPN"], "Province": ["-", "Tokyo"]},
                {"ISO3": ["JPN", "JPN"], "Province": ["-", "Tokyo"], "City": ["-", "-"]}
            ),
            (
                "ISO3",
                ["ISO3", "Province"],
                {"ISO3": ["JPN", "JPN"], "Prefecture": ["-", "Tokyo"]},
                {"ISO3": ["JPN", "JPN"], "Province": ["-", "Tokyo"]}
            ),
            (
                "ISO3",
                ["ISO3", "Province", "City"],
                {"ISO3": ["JPN", "JPN"], "City": ["-", "Chiyoda"]},
                {"ISO3": ["JPN", "JPN"], "Province": ["-", "-"], "City": ["-", "Chiyoda"]}
            ),
            (
                "ISO3",
                ["ISO3", "Province"],
                {"ISO3": ["JPN", "JPN", "JPN"], "Province": ["-", "Tokyo", "Tokyo"], "City": ["-", "-", "Chiyoda"]},
                {"ISO3": ["JPN", "JPN", "-"], "Province": ["-", "Tokyo", "-"]}
            ),
            (
                "ISO3",
                ["ISO3", "Province", "City"],
                {"Province": ["-", "Tokyo"], "City": ["-", "Chiyoda"]},
                {"ISO3": ["-", "-"], "Province": ["-", "Tokyo"], "City": ["-", "Chiyoda"]}
            ),
            (
                "Country",
                ["Country", "Province", "City"],
                {"Country": ["Japan", "Japan"], "Region": ["-", "Kanto"], "City": ["-", "Chiyoda"]},
                {"Country": ["JPN", "JPN"], "Province": ["-", "Kanto"], "City": ["-", "Chiyoda"]}
            ),
            (
                "Country",
                ["Country", "Province", "City"],
                {"Country": ["Japan", "Japan"], "Region": ["-", "Kanto"],
                    "Province": ["-", "Tokyo"], "City": ["-", "Chiyoda"]},
                {"Country": ["JPN", "JPN"], "Province": ["-", "Tokyo"], "City": ["-", "Chiyoda"]}
            ),
        )
    )
    def test_register(self, country, layers, data_dict, all_dict):
        day0, day1 = pd.to_datetime("2022-01-01"), pd.to_datetime("2022-01-02")
        raw = pd.concat([pd.DataFrame(data_dict), pd.DataFrame(data_dict)], axis=0, ignore_index=True)
        raw["date"] = [day0 for _ in range(len(raw) // 2)] + [day1 for _ in range(len(raw) // 2)]
        raw["Confirmed"] = np.arange(len(raw))
        raw["Recovered"] = 0
        collector = LayerAdjuster(layers=layers, country=country)
        with pytest.raises(ValueError):
            collector.register(
                data=raw, date="date", data_layers=["Country", "Country"], variables=["Confirmed", "Recovered"], citations="Manual")
        collector.register(
            data=raw, date="date", data_layers=list(data_dict.keys()), variables=["Confirmed"], citations="Manual")
        # All data
        all_df = pd.concat([pd.DataFrame(all_dict), pd.DataFrame(all_dict)], axis=0, ignore_index=True)
        all_df[Term.DATE] = [day0 for _ in range(len(all_df) // 2)] + [day1 for _ in range(len(all_df) // 2)]
        all_df["Confirmed"] = np.arange(len(all_df))
        all_df = all_df.sort_values([*layers, Term.DATE], ignore_index=True)
        assert_frame_equal(collector.all(variables=["Confirmed"]), all_df, check_dtype=False)
        assert collector.citations() == ["Manual"]
