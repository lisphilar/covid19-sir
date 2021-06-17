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
loader = cs.DataLoader("input")
jhu_data = loader.jhu()
oxcgrt_data = loader.oxcgrt()
# Scenario analysis
snl = cs.Scenario(country="Country name")
snl.register(jhu_data, extras=[oxcgrt_data])
snl.trend()
snl.estimate(cs.SIRF)
snl.fit()
snl.predict()
```

## Outputs
<!--dataframe, figures, stdout.-->

## Environment

- CovsirPhy version: 
- Python version: 
- Installation: poetry/pipenv/conda/pip
- System: WSL (Ubuntu)/Windows/Linux/Mac/Kaggle Notebook/Google Colaboratory
