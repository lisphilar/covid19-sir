
<img src="./docs/logo/covsirphy_headline.png" width="390" alt="CovsirPhy: COVID-19 analysis with phase-dependent SIRs">

[![PyPI version](https://badge.fury.io/py/covsirphy.svg)](https://badge.fury.io/py/covsirphy)
[![Downloads](https://pepy.tech/badge/covsirphy)](https://pepy.tech/project/covsirphy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/covsirphy)](https://badge.fury.io/py/covsirphy)
[![Build Status](https://semaphoreci.com/api/v1/lisphilar/covid19-sir/branches/master/shields_badge.svg)](https://semaphoreci.com/lisphilar/covid19-sir)  
[![GitHub license](https://img.shields.io/github/license/lisphilar/covid19-sir)](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE)
[![Maintainability](https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/maintainability)](https://codeclimate.com/github/lisphilar/covid19-sir/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/test_coverage)](https://codeclimate.com/github/lisphilar/covid19-sir/test_coverage)
[![Open Source Helpers](https://www.codetriage.com/lisphilar/covid19-sir/badges/users.svg)](https://www.codetriage.com/lisphilar/covid19-sir)

# CovsirPhy introduction

[<strong>Documentation</strong>](https://lisphilar.github.io/covid19-sir/index.html)
| [<strong>Installation</strong>](https://lisphilar.github.io/covid19-sir/INSTALLATION.html)
| [<strong>Quickest usage</strong>](https://lisphilar.github.io/covid19-sir/usage_quickest.html)
| [<strong>API reference</strong>](https://lisphilar.github.io/covid19-sir/covsirphy.html)
| [<strong>GitHub</strong>](https://github.com/lisphilar/covid19-sir)
| [<strong>Qiita (Japanese)</strong>](https://qiita.com/tags/covsirphy)

<strong>CovsirPhy is a Python library for COVID-19 (Coronavirus disease 2019) data analysis with phase-dependent SIR-derived ODE models. We can download datasets and analyse them easily. Scenario analysis with CovsirPhy enables us to make data-informed decisions. Please refer to "Method" part of [Kaggle Notebook: COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) to understand the methods.</strong>

<img src="./docs/gif/covsirphy_demo.gif" width="600">

## Functionalities
- [Data preparation and data visualization](https://lisphilar.github.io/covid19-sir/usage_dataset.html)
- [Phase setting with S-R Trend analysis](https://lisphilar.github.io/covid19-sir/usage_phases.html)
- [Numerical simulation of ODE models](https://lisphilar.github.io/covid19-sir/usage_theoretical.html)
    - Stable: SIR, SIR-D and SIR-F model
    - Development: SIR-FV and SEWIR-F model
- [Phase-dependent parameter estimation of ODE models](https://lisphilar.github.io/covid19-sir/usage_quickest.html)
- [Scenario analysis](https://lisphilar.github.io/covid19-sir/usage_quick.html): Simulate the number of cases with user-defined parameter values
- [(In development): Find the relationship of government response and parameter values](https://lisphilar.github.io/covid19-sir/usage_policy.html)

## Inspiration
- Monitor the spread of COVID-19
- Keep track parameter values/reproduction number in each country/province
- Find the relationship of reproductive number and measures taken by each country

<strong>If you have ideas or need new functionalities, please join this project.
Any suggestions with [Github Issues](https://github.com/lisphilar/covid19-sir/issues/new/choose) are always welcomed. Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.</strong>

## Installation
The latest stable version of CovsirPhy is available at [PyPI (The Python Package Index): covsirphy](https://pypi.org/project/covsirphy/) and supports Python 3.7 or newer versions. Details are explained in [Documentation: Installation](https://lisphilar.github.io/covid19-sir/INSTALLATION.html).

```
pip install --upgrade covsirphy
```

## Usage
Quickest tour of CovsirPhy is here. The following codes analyze the records in Japan, but we can change the country name when creating `Scenario` class instance for your own analysis.

```Python
import covsirphy as cs
# Download and update datasets
data_loader = cs.DataLoader("input")
jhu_data = data_loader.jhu()
population_data = data_loader.population()
# Check records
snl = cs.Scenario(jhu_data, population_data, country="Japan")
snl.records()
# S-R trend analysis
snl.trend().summary()
# Parameter estimation of SIR-F model
snl.estimate(cs.SIRF)
# History of reproduction number
_ = snl.history(target="Rt")
# History of parameters
_ = snl.history_rate()
_ = snl.history(target="rho")
# Simulation for 30 days
snl.add(days=30)
_ = snl.simulate()
```

Further information:

- [CovsirPhy documentation](https://lisphilar.github.io/covid19-sir/index.html)
- Example scripts in ["example" directory of this repository](https://github.com/lisphilar/covid19-sir/tree/master/example)
- [Kaggle: COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)


## Support
Please support this project as a developer (or a backer).
[![Become a backer](https://opencollective.com/covsirphy/tiers/backer.svg?avatarHeight=36&width=600)](https://opencollective.com/covsirphy)


## License: Apache License 2.0
Please refer to [LICENSE](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE) file.

## Citation
We have no original papers the author and contributors wrote, but please cite this package as follows.

CovsirPhy Development Team (2020), CovsirPhy, Python package for COVID-19 analysis with SIR-derived ODE models, https://github.com/lisphilar/covid19-sir

If you want to use SIR-F/SIR-FV/SEWIR-F model, S-R trend analysis, phase-dependent approach to SIR-derived models, and other scientific method performed with CovsirPhy, please cite the next Kaggle notebook.

Lisphilar (2020), Kaggle notebook, COVID-19 data with SIR model, https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

## Related work

Reproduction number evolution in each country:  
Ilyass Tabiai and Houda Kaddioui (2020), GitHub pages, COVID19 R0 tracker, https://ilylabs.github.io/projects/COVID-trackers/
