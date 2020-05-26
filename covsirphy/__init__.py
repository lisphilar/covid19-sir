#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import better_exceptions
from covsirphy.analysis import ODESimulator, ChangeFinder
from covsirphy.analysis import PhaseSeries, Scenario
from covsirphy.cleaning import CleaningBase, Word, Population
from covsirphy.cleaning import JHUData, CountryData
from covsirphy.ode import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF
from covsirphy.phase import PhaseData, NondimData, Estimator
from covsirphy.phase import SRData, Trend
from covsirphy.util import line_plot, jpn_map


__all__ = [
    "ODESimulator", "ChangeFinder",
    "PhaseSeries", "Scenario",
    "CleaningBase", "Word", "Population",
    "JHUData", "CountryData",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "PhaseData", "NondimData", "Estimator", "SRData", "Trend",
    "line_plot", "jpn_map",
]

# Check duplication
dup_list = [k for (k, v) in Counter(__all__).items() if v > 1]
if dup_list:
    dup_str = ', '.join(dup_list)
    raise Exception(f"Duplication was found in modules. {dup_str}")

# Show excetions in better format
better_exceptions.MAX_LENGTH = None
better_exceptions.hook()
