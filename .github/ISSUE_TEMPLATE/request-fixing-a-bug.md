---
name: Request fixing a bug
about: Customized issue template for fixing a bug.
title: '[Fix] '
labels: bug
assignees: ''

---

## Summary


## Codes
```Python
import covsirphy as cs
# Dataset preparation
data_loader = cs.DataLoader("input")
jhu_data = data_loader.jhu()
population_data = data_loader.population()
# Scenario analysis
snl = cs.Scenario(jhu_data, population_data, "Country name used")
```

## Outputs
(dataframe, figures, stdout.)

## Environment
- CovsirPhy version: 
- Python version: 3.8.5
- Installation: poetry/pipenv/conda/pip
- System: WSL (Ubuntu)/Windows/Linux/Mac/Kaggle Notebook/Google Colaboratory
