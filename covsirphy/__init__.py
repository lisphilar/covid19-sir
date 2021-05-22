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
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.error import deprecate
from covsirphy.util.error import SubsetNotFoundError, ScenarioNotFoundError, UnExecutedError
from covsirphy.util.error import PCRIncorrectPreconditionError, NotInteractiveError, UnExpectedValueError
from covsirphy.util.error import NotRegisteredMainError, NotRegisteredExtraError
from covsirphy.util.error import UnExpectedReturnValueError
from covsirphy.util.filer import save_dataframe
from covsirphy.util.argument import find_args
from covsirphy.util.filer import Filer
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.term import Term, Word
# visualization
from covsirphy.visualization.vbase import VisualizeBase
from covsirphy.visualization.colored_map import ColoredMap
from covsirphy.visualization.japan_map import jpn_map
from covsirphy.visualization.line_plot import LinePlot, line_plot
from covsirphy.visualization.bar_plot import BarPlot, bar_plot
from covsirphy.visualization.compare_plot import ComparePlot, compare_plot
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
# trend
from covsirphy.trend.trend_detector import TrendDetector, Trend, ChangeFinder
from covsirphy.trend.trend_plot import TrendPlot, trend_plot, line_plot_multiple
# ode
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.sir import SIR
from covsirphy.ode.sird import SIRD
from covsirphy.ode.sirf import SIRF
from covsirphy.ode.sirfv import SIRFV
from covsirphy.ode.sewirf import SEWIRF
from covsirphy.ode.ode_handler import ODEHandler
# simulation
from covsirphy.simulation.estimator import Estimator, Optimizer
from covsirphy.simulation.simulator import ODESimulator
# phase
from covsirphy.phase.phase_unit import PhaseUnit
from covsirphy.phase.phase_series import PhaseSeries
from covsirphy.phase.phase_estimator import MPEstimator
# regression
from covsirphy.regression.reg_handler import RegressionHandler
# analysis
from covsirphy.analysis.example_data import ExampleData
from covsirphy.analysis.data_handler import DataHandler
from covsirphy.analysis.param_tracker import ParamTracker
from covsirphy.analysis.phase_tracker import PhaseTracker
from covsirphy.analysis.scenario import Scenario
from covsirphy.analysis.model_validator import ModelValidator
# worldwide
from covsirphy.worldwide.policy import PolicyMeasures


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
    # util
    "StopWatch", "deprecate", "find_args", "Term", "Filer", "Evaluator",
    "SubsetNotFoundError", "ScenarioNotFoundError", "UnExecutedError",
    "PCRIncorrectPreconditionError", "NotInteractiveError",
    "UnExpectedValueError", "NotRegisteredMainError", "NotRegisteredExtraError",
    "UnExpectedReturnValueError",
    # visualization
    "VisualizeBase", "ColoredMap", "LinePlot", "line_plot", "BarPlot", "bar_plot",
    "ComparePlot", "compare_plot",
    # cleaning
    "CleaningBase", "DataLoader", "COVID19DataHub",
    "JHUData", "CountryData", "PopulationData", "OxCGRTData", "VaccineData",
    "PopulationPyramidData", "LinelistData", "PCRData", "JapanData", "JHUDataComplementHandler",
    # trend
    "TrendDetector", "TrendPlot", "trend_plot",
    # ode
    "ModelBase", "SIR", "SIRD", "SIRF", "SEWIRF", "ODEHandler",
    # regression
    "RegressionHandler",
    # analysis
    "ExampleData", "Scenario", "ModelValidator", "DataHandler", "PhaseTracker",
    # Deprecated
    "Population", "Word", "jpn_map", "SIRFV", "line_plot_multiple", "ChangeFinder", "Trend",
    "Optimizer", "save_dataframe", "PolicyMeasures", "ODESimulator", "Estimator", "ParamTracker",
    "PhaseSeries", "PhaseUnit", "MPEstimator",
]

# Show exceptions in better format if used from command line
if not hasattr(sys, "ps1") or not better_exceptions_installed:
    better_exceptions.MAX_LENGTH = None
    better_exceptions.hook()
