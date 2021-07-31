
<img src="https://raw.githubusercontent.com/lisphilar/covid19-sir/master/docs/logo/covsirphy_headline.png" width="390" alt="CovsirPhy: COVID-19 analysis with phase-dependent SIRs">

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

<strong>CovsirPhy is a Python library for COVID-19 (Coronavirus disease 2019) data analysis with phase-dependent SIR-derived ODE models. We can download datasets and analyse them easily. Scenario analysis with CovsirPhy enables us to make data-informed decisions. </strong>

<img src="https://raw.githubusercontent.com/lisphilar/covid19-sir/master/docs/gif/covsirphy_demo.gif" width="600">

## Functionalities

- [Data preparation and data visualization](https://lisphilar.github.io/covid19-sir/usage_dataset.html)
- [Phase setting with S-R Trend analysis](https://lisphilar.github.io/covid19-sir/usage_phases.html)
- [Numerical simulation of ODE models](https://lisphilar.github.io/covid19-sir/usage_theoretical.html): SIR, SIR-D and SIR-F model
- [Phase-dependent parameter estimation of ODE models](https://lisphilar.github.io/covid19-sir/usage_quickest.html)
- [Scenario analysis](https://lisphilar.github.io/covid19-sir/usage_quick.html): Simulate the number of cases with user-defined parameter values
- [Predict the parameter valuse to forecast the number of cases.](https://lisphilar.github.io/covid19-sir/usage_quick.html#Short-term-prediction-of-parameter-values)

## Inspiration

- Monitor the spread of COVID-19
- Keep track parameter values/reproduction number in each country/province
- Find the relationship of reproductive number and measures taken by each country

<strong>If you have ideas or need new functionalities, please join this project.
Any suggestions with [Github Issues](https://github.com/lisphilar/covid19-sir/issues/new/choose) are always welcomed. Questions are also great. Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.</strong>

## Installation

The latest stable version of CovsirPhy is available at [PyPI (The Python Package Index): covsirphy](https://pypi.org/project/covsirphy/) and supports Python 3.7 or newer versions. Details are explained in [Documentation: Installation](https://lisphilar.github.io/covid19-sir/INSTALLATION.html).

```bash
pip install --upgrade covsirphy
```

## Usage

Quickest tour of CovsirPhy is here. The following codes analyze the records in Japan, but we can change the country name when creating `Scenario` class instance for your own analysis.

```Python
import covsirphy as cs
# Download and update datasets
data_loader = cs.DataLoader("input")
jhu_data = data_loader.jhu()
# Select country name and register the data
snl = cs.Scenario(country="Japan")
snl.register(jhu_data)
# Check records
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
- [Kaggle: COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)

## Release notes

Release notes are [here](https://github.com/lisphilar/covid19-sir/releases). Titles & links of issues are listed with acknowledgement.

We can see the release plan for the next stable version in [milestone page of the GitHub repository](https://github.com/lisphilar/covid19-sir/milestones). If you find a highly urgent matter, please let us know via [issue page](https://github.com/lisphilar/covid19-sir/issues).

## Support

Please support this project as a developer (or a backer).
[![Become a backer](https://opencollective.com/covsirphy/tiers/backer.svg?avatarHeight=36&width=600)](https://opencollective.com/covsirphy)

## Developers

CovsirPhy library is developed by a community of volunteers. Please see the full list [here](https://github.com/lisphilar/covid19-sir/graphs/contributors).

This project started in Kaggle platform. Lisphilar published [Kaggle Notebook: COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) on 12Feb2020 and developed it, discussing with Kaggle community. On 07May2020, "covid19-sir" repository was created. On 10May2020, `covsirphy` version 1.0.0 was published in GitHub. First release in PyPI (version 2.3.0) was on 28Jun2020.

## License: Apache License 2.0

Please refer to [LICENSE](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE) file.

## Citation

We have no original papers the author and contributors wrote, but please cite this library as follows with version number (`import covsirphy as cs; cs.__version__`).

CovsirPhy Development Team (2020-2021), CovsirPhy version [version number]: Python library for COVID-19 analysis with phase-dependent SIR-derived ODE models, [https://github.com/lisphilar/covid19-sir](https://github.com/lisphilar/covid19-sir)

If you want to use SIR-F model, S-R trend analysis, phase-dependent approach to SIR-derived models, and other scientific method performed with CovsirPhy, please cite the next Kaggle notebook.

Hirokazu Takaya (2020-2021), Kaggle Notebook, COVID-19 data with SIR model, [https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)

We can check the citation with the following script.

```Python
import covsirphy as cs
cs.__citation__
```
