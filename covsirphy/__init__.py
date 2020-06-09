#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import sys
import better_exceptions
from covsirphy.analysis import ODESimulator, ChangeFinder
from covsirphy.analysis import PhaseSeries, Scenario
from covsirphy.cleaning import CleaningBase, Word, Population
from covsirphy.cleaning import JHUData, CountryData
from covsirphy.ode import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF
from covsirphy.phase import PhaseData, NondimData, Estimator
from covsirphy.phase import SRData, Trend
from covsirphy.util import line_plot, jpn_map, StopWatch


__all__ = [
    "ODESimulator", "ChangeFinder",
    "PhaseSeries", "Scenario",
    "CleaningBase", "Word", "Population",
    "JHUData", "CountryData",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "PhaseData", "NondimData", "Estimator", "SRData", "Trend",
    "line_plot", "jpn_map", "StopWatch",
]

# Check duplication
dup_list = [k for (k, v) in Counter(__all__).items() if v > 1]
if dup_list:
    dup_str = ', '.join(dup_list)
    raise Exception(f"Duplication was found in modules. {dup_str}")

# Show exceptions in better format if used from command line
if not hasattr(sys, "ps1"):
    better_exceptions.MAX_LENGTH = None
    better_exceptions.hook()
