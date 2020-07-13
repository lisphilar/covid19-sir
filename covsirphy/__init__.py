#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import sys
try:
    import better_exceptions
    better_exceptions_installed = True
except ImportError:
    better_exceptions_installed = False
from covsirphy.__version__ import __version__
from covsirphy.analysis import ODESimulator, ChangeFinder
from covsirphy.analysis import PhaseSeries, Scenario
from covsirphy.cleaning import Term, CleaningBase, DataLoader
from covsirphy.cleaning import JHUData, ExampleData
from covsirphy.cleaning import CountryData, PopulationData, OxCGRTData
from covsirphy.ode import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF
from covsirphy.phase import Estimator, Trend
from covsirphy.util import line_plot, jpn_map, StopWatch, error
# Deprecated
from covsirphy.cleaning import Population, Word


def get_version():
    """
    Return the version number, like CovsirPhy v0.0.0
    """
    return f"CovsirPhy v{__version__}"


__all__ = [
    "ODESimulator", "ChangeFinder",
    "PhaseSeries", "Scenario",
    "Term", "CleaningBase", "DataLoader",
    "JHUData", "ExampleData", "CountryData", "PopulationData", "OxCGRTData",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "Estimator", "Trend",
    "line_plot", "jpn_map", "StopWatch", "error",
    # Deprecated
    "Population", "Word",
]

# Check duplication
dup_list = [k for (k, v) in Counter(__all__).items() if v > 1]
if dup_list:
    dup_str = ', '.join(dup_list)
    raise Exception(f"Duplication was found in modules. {dup_str}")

# Show exceptions in better format if used from command line
if not hasattr(sys, "ps1") or not better_exceptions_installed:
    better_exceptions.MAX_LENGTH = None
    better_exceptions.hook()
