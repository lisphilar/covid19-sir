#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.util.error import deprecate
from covsirphy._deprecated._mbase import ModelBase


class SIRFV(ModelBase):
    """
    SIR-FV model.

    Args:
        population (int): total population
            theta (float)
            kappa (float)
            rho (float)
            sigma (float)
            omega (float) or v_per_day (int)
    """
    # Model name
    NAME = "SIR-FV"
    # names of parameters
    PARAMETERS = ["theta", "kappa", "rho", "sigma", "omega"]
    DAY_PARAMETERS = [
        "alpha1 [-]", "1/alpha2 [day]", "1/beta [day]", "1/gamma [day]",
        "Vaccinated [persons]"
    ]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F,
        "v": ModelBase.V
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([0, 10, 10, 2, 0])
    # Variables that increases monotonically
    VARS_INCREASE = [ModelBase.R, ModelBase.F]
    # Example set of parameters and initial values
    EXAMPLE = {
        ModelBase.STEP_N: 180,
        ModelBase.N.lower(): 1_000_000,
        ModelBase.PARAM_DICT: {
            "theta": 0.002, "kappa": 0.005, "rho": 0.2, "sigma": 0.075,
            "omega": 0.001,
        },
        ModelBase.Y0_DICT: {
            ModelBase.S: 999_000, ModelBase.CI: 1000, ModelBase.R: 0, ModelBase.F: 0,
            ModelBase.V: 0,
        },
    }

    @deprecate(old="ModelBase", new="ODEModel", version="2.24.0-xi")
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SIR-FV model was removed because vaccinated persons may move "
            "to the other compartments. Please use SIR-F model and adjust parameter values "
            "of SIR-F model, considering the impact of vaccinations on infectivity, effectivity and safety."
        )
