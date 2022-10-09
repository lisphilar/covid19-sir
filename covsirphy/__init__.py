#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from covsirphy.util.filer import Filer
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
# visualization
from covsirphy.visualization.vbase import VisualizeBase
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
# dynamics
from covsirphy.dynamics.ode import ODEModel
from covsirphy.dynamics.sir import SIRModel
from covsirphy.dynamics.sird import SIRDModel
from covsirphy.dynamics.sirf import SIRFModel
from covsirphy.dynamics.sewirf import SEWIRFModel
from covsirphy.dynamics.dynamics import Dynamics
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
]
