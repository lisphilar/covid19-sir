#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Demonstration
# Use this codes in the top directory
import covsirphy as cs

# Load datasets
data_loader = cs.DataLoader("input")
jhu_data = data_loader.jhu(verbose=False)
population_data = data_loader.population(verbose=False)

# Set country name
scenario = cs.Scenario(jhu_data, population_data, country="Italy")

# Show records
_ = scenario.records()

# Scenario analysis
scenario.trend()
scenario.summary()

# Parameter estimation with SIR-F model
scenario.estimate(cs.SIRF)
scenario.param_history(targets=["Rt"], divide_by_first=False).T

# Simulation
scenario.add_phase(end_date="01Jan2021")
_ = scenario.simulate()
