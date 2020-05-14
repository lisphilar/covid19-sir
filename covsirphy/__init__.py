#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from covsirphy.analysis import Estimator, Predicter, Scenario
from covsirphy.analysis import simulation, Trend
from covsirphy.cleaning import CleaningBase, JHUData, Word, Population, CountryData
from covsirphy.ode import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF
from covsirphy.selection import select_area, SelectArea, create_target_df
from covsirphy.util import line_plot, jpn_map


__all__ = [
    "Estimator", "Predicter", "Scenario",
    "simulation", "Trend",
    "CleaningBase", "JHUData", "Word", "Population", "CountryData",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "select_area", "SelectArea", "create_target_df",
    "line_plot", "jpn_map",
]

# Check duplication
dup_list = [k for (k, v) in Counter(__all__).items() if v > 1]
if dup_list:
    dup_str = ', '.join(dup_list)
    raise Exception(f"Duplication was found in modules. {dup_str}")
