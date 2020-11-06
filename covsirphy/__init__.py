#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
try:
    import better_exceptions
    better_exceptions_installed = True
except ImportError:
    better_exceptions_installed = False
# version
from covsirphy.__version__ import __version__
# util
from covsirphy.util.plotting import line_plot
from covsirphy.util.map import jpn_map
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.error import deprecate
from covsirphy.util.file import save_dataframe
from covsirphy.util.argument import find_args
# cleaning
from covsirphy.cleaning.term import Term, Word
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.population import PopulationData, Population
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.covid19datahub import COVID19DataHub
from covsirphy.cleaning.dataloader import DataLoader
# ode
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.sir import SIR
from covsirphy.ode.sird import SIRD
from covsirphy.ode.sirf import SIRF
from covsirphy.ode.sirfv import SIRFV
from covsirphy.ode.sewirf import SEWIRF
# simulation
from covsirphy.simulation.estimation_study import EstimationStudy
from covsirphy.simulation.estimator import Estimator
from covsirphy.simulation.optimize import Optimizer
from covsirphy.simulation.simulator import ODESimulator
# phase
from covsirphy.phase.trend import Trend
from covsirphy.phase.sr_change import ChangeFinder
from covsirphy.phase.phase_unit import PhaseUnit
from covsirphy.phase.phase_series import PhaseSeries
from covsirphy.phase.phase_estimator import MPEstimator
# analysis
from covsirphy.analysis.example_data import ExampleData
from covsirphy.analysis.scenario import Scenario
from covsirphy.analysis.model_validator import ModelValidator
# worldwide
from covsirphy.worldwide.policy import PolicyMeasures


def get_version():
    """
    Return the version number, like CovsirPhy v0.0.0
    """
    return f"CovsirPhy v{__version__}"


__all__ = [
    "ExampleData", "Scenario", "ModelValidator",
    "ODESimulator", "ChangeFinder",
    "PhaseSeries", "PhaseUnit", "MPEstimator",
    "Term", "CleaningBase", "DataLoader", "COVID19DataHub",
    "JHUData", "CountryData", "PopulationData", "OxCGRTData",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "Estimator", "Trend", "Optimizer", "EstimationStudy",
    "line_plot", "jpn_map", "StopWatch", "deprecate", "find_args",
    "save_dataframe",
    "PolicyMeasures",
    # Deprecated
    "Population", "Word",
]

# Show exceptions in better format if used from command line
if not hasattr(sys, "ps1") or not better_exceptions_installed:
    better_exceptions.MAX_LENGTH = None
    better_exceptions.hook()
