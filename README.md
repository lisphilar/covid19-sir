
<img src="./docs/logo/covsirphy_headline.png" width="390" alt="CovsirPhy: COVID-19 analysis with phase-dependent SIRs">

[![PyPI version](https://badge.fury.io/py/covsirphy.svg)](https://badge.fury.io/py/covsirphy)
[![Downloads](https://pepy.tech/badge/covsirphy)](https://pepy.tech/project/covsirphy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/covsirphy)](https://badge.fury.io/py/covsirphy)
[![Build Status](https://semaphoreci.com/api/v1/lisphilar/covid19-sir/branches/master/shields_badge.svg)](https://semaphoreci.com/lisphilar/covid19-sir)  
[![GitHub license](https://img.shields.io/github/license/lisphilar/covid19-sir)](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE)
[![Maintainability](https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/maintainability)](https://codeclimate.com/github/lisphilar/covid19-sir/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/test_coverage)](https://codeclimate.com/github/lisphilar/covid19-sir/test_coverage)


# CovsirPhy introduction

[<strong>Documentation</strong>](https://lisphilar.github.io/covid19-sir/index.html)
| [<strong>Installation</strong>](https://lisphilar.github.io/covid19-sir/INSTALLATION.html)
| [<strong>Quickest usage</strong>](https://lisphilar.github.io/covid19-sir/usage_quickest.html)
| [<strong>API reference</strong>](https://lisphilar.github.io/covid19-sir/covsirphy.html)
| [<strong>Qiita (Japanese)</strong>](https://qiita.com/tags/covsirphy)

<strong>CovsirPhy is a Python package for COVID-19 (Coronavirus disease 2019) data analysis with phase-dependent SIR-derived ODE models. We can download datasets and analyse it easily. This will be a helpful tool for data-informed decision making. Please refer to "Method" part of [Kaggle Notebook: COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) to understand the methods.</strong>

<img src="./docs/gif/covsirphy_demo.gif" width="600">

## Functionalities
- Data preparation and data visualization
- [Phase setting with S-R Trend analysis](https://lisphilar.github.io/covid19-sir/usage_phases.html)
- Numerical simulation of ODE models
    - Stable: SIR, SIR-D and SIR-F model
    - Development: SIR-FV and SEWIR-F model
- Phase-dependent parameter estimation of ODE models
- Scenario analysis: Simulate the number of cases with user-defined parameter values
- (In development): Find the relationship of government response and parameter values

## Inspiration
- Monitor the spread of COVID-19
- Keep track parameter values/reproduction number in each country/province
- Find the relationship of reproductive number and measures taken by each country

<strong>If you have ideas or need new functionalities, please join this project.
Any suggestions with [Github Issues](https://github.com/lisphilar/covid19-sir/issues/new/choose) are always welcomed. Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.</strong>

## Installation and dataset preparation
We have the following options to start analysis with CovsirPhy. Datasets are not included in this package, but we can prepare them with `DataLoader` class.

|                            |Installation     |Dataset preparation                          |
|:---------------------------|:----------------|:--------------------------------------------|
|Standard users              |pip/pipenv       |Automated with `DataLoader` class            |
|Developers                  |git-cloning      |Automated with `DataLoader` class            |
|Kagglers (local environment)|git-cloning      |Kaggle API and Python script and `DataLoader`|
|Kagglers (Kaggle platform)  |pip              |Kaggle Datasets and `DataLoader`             |

<strong>[Installation and dataset preparation](https://lisphilar.github.io/covid19-sir/INSTALLATION.html) explains how to install and prepare datasets for all users.</strong>

Stable versions of Covsirphy are available at [PyPI (The Python Package Index): covsirphy](https://pypi.org/project/covsirphy/) and support Python 3.6 or newer versions.
```
pip install covsirphy --upgrade
```

Development versions are in [GitHub repository: CovsirPhy](https://github.com/lisphilar/covid19-sir).
```
pip install "git+https://github.com/lisphilar/covid19-sir.git#egg=covsirphy"
```

Main datasets will be retrieved via [COVID-19 Data Hub](https://covid19datahub.io/) and the citation is  
Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub", Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.


## Usage
Quickest tour of CovsirPhy is here. The following codes analyze the records in Japan, but we can change the country name when creating `Scenario` class instance for your own analysis.

```Python
import covsirphy as cs
# Download datasets
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

- [Quickest version](https://lisphilar.github.io/covid19-sir/usage_quickest.html)
- [Quick version](https://lisphilar.github.io/covid19-sir/usage_quick.html)
- [Details: phases](https://lisphilar.github.io/covid19-sir/usage_phases.html)
- [Details: theoretical datasets](https://lisphilar.github.io/covid19-sir/usage_theoretical.html)
- [Details: policy measures](https://lisphilar.github.io/covid19-sir/usage_policy.html)
- Example codes in ["example" directory of this repository](https://github.com/lisphilar/covid19-sir/tree/master/example)
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
