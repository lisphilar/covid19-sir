#!/bin/bash

# Initialize the input directory
mkdir -p input
rm input/*.csv 2>/dev/null

# Download datasets from Kaggle
pipenv run ./input.py
