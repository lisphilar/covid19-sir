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

    def __init__(self, jhu_data, population_data, oxcgrt_data, model, tau=None):
        # Records
        self.jhu_data = self.ensure_instance(
            jhu_data, JHUData, name="jhu_data")
        # Population
        self.population_data = self.ensure_instance(
            population_data, PopulationData, name="population_data")
        # OxCGRT
        self.oxcgrt_data = self.ensure_instance(
            oxcgrt_data, OxCGRTData, name="oxcgrt_data")
        # Model
        self.model = self.ensure_subclass(model, ModelBase)
        # tau value must be shared
        self.tau = self.ensure_natural_int(tau, name="tau")

    def countries(self):
        """
        Return names of countries where records are registered.

        Returns:
            (list[str]): list of country names
        """
        j_list = self.jhu_data.countries()
        p_list = self.population_data.countries()
        o_list = self.oxcgrt_data.countries()
        return list(set(j_list) & set(p_list) & set(o_list))
