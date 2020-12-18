---
name: Request fixing a bug
about: Customized issue template for fixing a bug.
title: '[Fix] '
labels: bug
assignees: ''

---

## Summary
Please edit this to explain the summary of the bug.
`` needs to return ..., but returns ...

## (Optional) Related classes
- `covsirphy.`

## Codes and outputs:
```Python
import covsirphy as cs
# Dataset preparation
data_loader = cs.DataLoader("input")
jhu_data = data_loader.jhu()
population_data = data_loader.population()
# Scenario analysis
snl = cs.Scenario(jhu_data, population_data, "Country name used")
```
This code returns 

## Environment
- CovsirPhy version: 
- Python version; 3.8
- Installation: pipenv
- System: WSL (Ubuntu)/Windows/Linux/Mac/Kaggle Notebook/Google Colaboratory
