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
from covsirphy.__citation__ import __citation__
# util
from covsirphy.util.config import config
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.error import deprecate, experimental
from covsirphy.util.error import ExperimentalWarning
from covsirphy.util.error import SubsetNotFoundError, ScenarioNotFoundError
from covsirphy.util.error import PCRIncorrectPreconditionError, NotInteractiveError
from covsirphy.util.error import NotRegisteredError, NotRegisteredMainError, NotRegisteredExtraError
from covsirphy.util.error import UnExpectedReturnValueError, UnExpectedNoneError
from covsirphy.util.error import DBLockedError, NotDBLockedError, NotNoneError, NotEnoughDataError
from covsirphy.util.error import AlreadyCalledError, NotIncludedError, NAFoundError, UnExecutedError, UnExpectedTypeError
from covsirphy.util.error import EmptyError, UnExpectedValueRangeError, UnExpectedValueError, NotSubclassError, UnExpectedLengthError
from covsirphy.util.alias import Alias
from covsirphy.util.filer import save_dataframe
from covsirphy.util.argument import find_args
from covsirphy.util.filer import Filer
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term, Word
# visualization
from covsirphy.visualization.vbase import VisualizeBase
from covsirphy._deprecated.colored_map import ColoredMap
from covsirphy._deprecated.japan_map import jpn_map
from covsirphy.visualization.line_plot import LinePlot, line_plot
from covsirphy.visualization.bar_plot import BarPlot, bar_plot
from covsirphy.visualization.compare_plot import ComparePlot, compare_plot
from covsirphy.visualization.scatter_plot import ScatterPlot, scatter_plot
# gis
from covsirphy.gis.gis import GIS
# downloading
from covsirphy.downloading.downloader import DataDownloader
# engineering
from covsirphy.engineering.engineer import DataEngineer
# cleaning
from covsirphy._deprecated.cbase import CleaningBase
from covsirphy._deprecated.jhu_data import JHUData
from covsirphy._deprecated.jhu_complement import JHUDataComplementHandler
from covsirphy._deprecated.country_data import CountryData
from covsirphy._deprecated.japan_data import JapanData
from covsirphy._deprecated.population import PopulationData, Population
from covsirphy._deprecated.pyramid import PopulationPyramidData
from covsirphy._deprecated.oxcgrt import OxCGRTData
from covsirphy._deprecated.pcr_data import PCRData
from covsirphy._deprecated.linelist import LinelistData
from covsirphy._deprecated.vaccine_data import VaccineData
from covsirphy._deprecated.mobility_data import MobilityData
# loading
from covsirphy._deprecated.covid19datahub import COVID19DataHub
from covsirphy._deprecated.dataloader import DataLoader
# trend
from covsirphy._deprecated.trend_detector import TrendDetector, Trend, ChangeFinder
from covsirphy._deprecated.trend_plot import TrendPlot, trend_plot, line_plot_multiple
# ode
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated._sir import SIR
from covsirphy._deprecated._sird import SIRD
from covsirphy._deprecated._sirf import SIRF
from covsirphy._deprecated._sirfv import SIRFV
from covsirphy._deprecated._sewirf import SEWIRF
from covsirphy._deprecated.ode_handler import ODEHandler
# dynamics
from covsirphy.dynamics.ode import ODEModel
from covsirphy.dynamics.sir import SIRModel
from covsirphy.dynamics.sird import SIRDModel
from covsirphy.dynamics.sirf import SIRFModel
from covsirphy.dynamics.sewirf import SEWIRFModel
from covsirphy.dynamics.dynamics import Dynamics
# simulation
from covsirphy._deprecated.estimator import Estimator, Optimizer
from covsirphy._deprecated.simulator import ODESimulator
# phase
from covsirphy._deprecated.phase_unit import PhaseUnit
from covsirphy._deprecated.phase_series import PhaseSeries
from covsirphy._deprecated.phase_estimator import MPEstimator
# regression
from covsirphy._deprecated.reg_handler import RegressionHandler
# automl
from covsirphy._deprecated.automl_handler import AutoMLHandler
# analysis
from covsirphy._deprecated.example_data import ExampleData
from covsirphy._deprecated.data_handler import DataHandler
from covsirphy._deprecated.param_tracker import ParamTracker
from covsirphy._deprecated.phase_tracker import PhaseTracker
from covsirphy._deprecated.scenario import Scenario
from covsirphy._deprecated.model_validator import ModelValidator
# worldwide
from covsirphy._deprecated.policy import PolicyMeasures
# science
from covsirphy.science.ml import MLEngineer
from covsirphy.science.ode_scenario import ODEScenario


def get_version():
    """
    Return the version number, like CovsirPhy v0.0.0

    Returns:
        str
    """
    return f"CovsirPhy v{__version__}"


def get_citation():
    """
    Return the citation of CovsirPhy

    Returns:
        str
    """
    return __citation__


__all__ = [
    # util-error
    "experimental", "ExperimentalWarning",
    "SubsetNotFoundError", "ScenarioNotFoundError",
    "PCRIncorrectPreconditionError", "NotInteractiveError",
    "NotRegisteredError", "NotRegisteredMainError", "NotRegisteredExtraError",
    "UnExpectedReturnValueError", "DBLockedError", "NotDBLockedError",
    "AlreadyCalledError", "NotIncludedError", "NAFoundError", "UnExecutedError", "UnExpectedTypeError",
    "EmptyError", "UnExpectedValueRangeError", "UnExpectedValueError", "NotSubclassError", "UnExpectedLengthError",
    "Validator", "UnExpectedNoneError", "NotNoneError", "NotEnoughDataError",
    # util
    "config", "StopWatch", "deprecate", "Term", "Filer", "Evaluator", "Alias",
    # visualization
    "VisualizeBase", "LinePlot", "line_plot", "BarPlot", "bar_plot",
    "ComparePlot", "compare_plot", "ScatterPlot", "scatter_plot",
    # gis
    "GIS",
    # downloading
    "DataDownloader",
    # engineer
    "DataEngineer",
    # dynamics
    "ODEModel", "SIRModel", "SIRDModel", "SIRFModel", "SEWIRFModel", "Dynamics",
    # science
    "MLEngineer", "ODEScenario",
    # Deprecated
    "Population", "Word", "jpn_map", "SIRFV", "line_plot_multiple", "ChangeFinder", "Trend",
    "Optimizer", "save_dataframe", "PolicyMeasures", "ODESimulator", "Estimator", "ParamTracker",
    "PhaseSeries", "PhaseUnit", "MPEstimator", "COVID19DataHub", "LinelistData", "PopulationData",
    "CountryData", "ColoredMap", "ExampleData", "ModelValidator", "find_args",
    "Scenario", "DataHandler", "PhaseTracker",
    "TrendDetector", "TrendPlot", "trend_plot",
    "ModelBase", "SIR", "SIRD", "SIRF", "SEWIRF", "ODEHandler",
    "RegressionHandler",
    "JHUData", "OxCGRTData", "VaccineData", "PCRData", "JHUDataComplementHandler", "MobilityData",
    "DataLoader", "AutoMLHandler", "JapanData", "CleaningBase", "PopulationPyramidData",
]

# Show exceptions in better format if used from command line
if not hasattr(sys, "ps1") or not better_exceptions_installed:
    better_exceptions.MAX_LENGTH = None
    better_exceptions.hook()
