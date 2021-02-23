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
from covsirphy.util.plotting import line_plot, line_plot_multiple
from covsirphy.util.optimize import Optimizer
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.error import deprecate
from covsirphy.util.error import SubsetNotFoundError, ScenarioNotFoundError, UnExecutedError
from covsirphy.util.error import PCRIncorrectPreconditionError, NotInteractiveError, UnExpectedValueError
from covsirphy.util.error import NotRegisteredMainError, NotRegisteredExtraError
from covsirphy.util.file import save_dataframe
from covsirphy.util.argument import find_args
from covsirphy.util.term import Term, Word
# visualization
from covsirphy.visualization.vbase import VisualizeBase
from covsirphy.visualization.colored_map import ColoredMap
from covsirphy.visualization.japan_map import jpn_map
# cleaning
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.jhu_complement import JHUDataComplementHandler
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.japan_data import JapanData
from covsirphy.cleaning.population import PopulationData, Population
from covsirphy.cleaning.pyramid import PopulationPyramidData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.pcr_data import PCRData
from covsirphy.cleaning.covid19datahub import COVID19DataHub
from covsirphy.cleaning.linelist import LinelistData
from covsirphy.cleaning.vaccine_data import VaccineData
from covsirphy.cleaning.dataloader import DataLoader
# ode
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.sir import SIR
from covsirphy.ode.sird import SIRD
from covsirphy.ode.sirf import SIRF
from covsirphy.ode.sirfv import SIRFV
from covsirphy.ode.sewirf import SEWIRF
# simulation
from covsirphy.simulation.estimator import Estimator
from covsirphy.simulation.simulator import ODESimulator
# phase
from covsirphy.phase.trend import Trend
from covsirphy.phase.sr_change import ChangeFinder
from covsirphy.phase.phase_unit import PhaseUnit
from covsirphy.phase.phase_series import PhaseSeries
from covsirphy.phase.phase_estimator import MPEstimator
# analysis
from covsirphy.analysis.example_data import ExampleData
from covsirphy.analysis.data_handler import DataHandler
from covsirphy.analysis.param_tracker import ParamTracker
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
    "ExampleData", "Scenario", "ModelValidator", "ParamTracker",
    "ODESimulator", "ChangeFinder", "DataHandler",
    "PhaseSeries", "PhaseUnit", "MPEstimator",
    "Term", "CleaningBase", "DataLoader", "COVID19DataHub",
    "JHUData", "CountryData", "PopulationData", "OxCGRTData", "VaccineData",
    "PopulationPyramidData",
    "LinelistData", "PCRData", "JapanData", "JHUDataComplementHandler",
    "ModelBase", "SIR", "SIRD", "SIRF", "SIRFV", "SEWIRF",
    "Estimator", "Trend", "Optimizer",
    "line_plot", "StopWatch", "deprecate", "find_args",
    "line_plot_multiple",
    "save_dataframe",
    "PolicyMeasures",
    "SubsetNotFoundError", "ScenarioNotFoundError", "UnExecutedError",
    "PCRIncorrectPreconditionError", "NotInteractiveError",
    "UnExpectedValueError", "NotRegisteredMainError", "NotRegisteredExtraError",
    "VisualizeBase", "ColoredMap",
    # Deprecated
    "Population", "Word", "jpn_map",
]

# Show exceptions in better format if used from command line
if not hasattr(sys, "ps1") or not better_exceptions_installed:
    better_exceptions.MAX_LENGTH = None
    better_exceptions.hook()
