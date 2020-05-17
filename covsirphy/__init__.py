#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from covsirphy.analysis import Simulator
from covsirphy.cleaning import CleaningBase, JHUData, Word, Population
from covsirphy.cleaning import CountryData
from covsirphy.ode import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF
from covsirphy.phase import NondimData, Estimator
from covsirphy.selection import select_area, SelectArea, create_target_df
from covsirphy.util import line_plot, jpn_map


__all__ = [
    "Simulator",
    "CleaningBase", "JHUData", "Word", "Population",
    "CountryData",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "NondimData", "Estimator",
    "select_area", "SelectArea", "create_target_df",
    "line_plot", "jpn_map",
]

# Check duplication
dup_list = [k for (k, v) in Counter(__all__).items() if v > 1]
if dup_list:
    dup_str = ', '.join(dup_list)
    raise Exception(f"Duplication was found in modules. {dup_str}")
