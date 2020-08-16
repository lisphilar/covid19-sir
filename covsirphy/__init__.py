#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
try:
    import better_exceptions
    better_exceptions_installed = True
except ImportError:
    better_exceptions_installed = False
from covsirphy.__version__ import __version__
from covsirphy.cleaning import Term, CleaningBase, DataLoader
from covsirphy.cleaning import JHUData, CountryData, PopulationData, OxCGRTData
from covsirphy.ode import ModelBase, SIR, SIRD, SIRF, SIRFV, SEWIRF
from covsirphy.phase import Trend, ChangeFinder, PhaseSeries, PhaseUnit, MPEstimator
from covsirphy.simulation import Optimizer, ODESimulator, Estimator
from covsirphy.util import line_plot, jpn_map, StopWatch, error, find_args
from covsirphy.analysis import ExampleData, Scenario
from covsirphy.worldwide import PolicyMeasures
# Deprecated
from covsirphy.cleaning import Population, Word


def get_version():
    """
    Return the version number, like CovsirPhy v0.0.0
    """
    return f"CovsirPhy v{__version__}"


__all__ = [
    "ODESimulator", "ChangeFinder",
    "PhaseSeries", "Scenario", "ExampleData", "PhaseUnit", "MPEstimator",
    "Term", "CleaningBase", "DataLoader",
    "JHUData", "CountryData", "PopulationData", "OxCGRTData",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "Estimator", "Trend", "Optimizer",
    "line_plot", "jpn_map", "StopWatch", "error", "find_args",
    "PolicyMeasures",
    # Deprecated
    "Population", "Word",
]

# Show exceptions in better format if used from command line
if not hasattr(sys, "ps1") or not better_exceptions_installed:
    better_exceptions.MAX_LENGTH = None
    better_exceptions.hook()
