#!/bin/bash
# all example codes will be done using pipenv
# Outputs will be saved in example/output directory

# Data cleaning
echo "<Data cleaning>"
echo "JHU data"
pipenv run python -m example.dataset_jhu
echo "Population"
pipenv run python -m example.dataset_population
echo "OxCGRT"
pipenv run python -m example.dataset_oxcgrt

# ODE simulation and hyperparameter estimation
echo "<ODE simulation and hyperparameter estimation>"
echo "SIR model"
pipenv run python -m example.sir_model
echo "SIR-D model"
pipenv run python -m example.sird_model
echo "SIR-F model"
pipenv run python -m example.sirf_model
echo "SIR-FV model"
pipenv run python -m example.sirfv_model
echo "SEWIR-F model"
pipenv run python -m example.sewirf_model

# Long ODE simulation with SIR-F model
echo "<Long ODE simulation with SIR-F model>"
pipenv run python -m example.long_simulation

# Scenario analysis
echo "<Scenario analysis>"
pipenv run python -m example.scenario_analysis
