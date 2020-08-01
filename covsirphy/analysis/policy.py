#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.analysis.scenario import Scenario


class PolicyMeasures(Term):
    """
    Analyse the relationship of policy measures and parameters of ODE models.
    This analysis will be done at country level because OxCGRT tracks policies at country level.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        model (covsirphy.ModelBase): ODE model
        tau (int or None): tau value
    """

    def __init__(self, jhu_data, population_data, model, tau=None):
        # Population
        population_data = self.ensure_instance(
            population_data, PopulationData, name="population_data")
        # Records
        self.jhu_data = self.ensure_instance(
            jhu_data, JHUData, name="jhu_data")
        # Model
        self.model = self.ensure_subclass(model, ModelBase)
        # tau value must be shared
        self.tau = self.ensure_natural_int(tau, name="tau")
