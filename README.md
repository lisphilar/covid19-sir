# CovsirPhy: COVID-19 analysis with SIRs
[![PyPI version](https://badge.fury.io/py/covsirphy.svg)](https://badge.fury.io/py/covsirphy)
[![Downloads](https://pepy.tech/badge/covsirphy)](https://pepy.tech/project/covsirphy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/covsirphy)](https://badge.fury.io/py/covsirphy)  
[![GitHub license](https://img.shields.io/github/license/lisphilar/covid19-sir)](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE)
[![Maintainability](https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/maintainability)](https://codeclimate.com/github/lisphilar/covid19-sir/maintainability)
[![test](https://github.com/lisphilar/covid19-sir/workflows/test/badge.svg)](https://github.com/lisphilar/covid19-sir/actions)

<strong>CovsirPhy is a Python package for COVID-19 (Coronavirus disease 2019) data analysis with SIR-derived ODE models. Please refer to "Method" part of [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) notebook in Kaggle to understand the methods.</strong>

<img src="./docs/gif/covsirphy_demo.gif" width="600">

## Functionalities
- Downloading and cleaning data: refer to "Installation and dataset preparation" section
- Data visualization
- S-R Trend analysis to determine the change points of parameters
- Numerical simulation of ODE models
- Description of ODE models
    - Basic class of ODE models
    - SIR, SIR-D, SIR-F, SIR-FV and SEWIR-F model
- Parameter Estimation of ODE models
- Scenario analysis: Simulate the number of cases with user-defined parameter values

## Inspiration
- Monitor the spread of COVID-19
- Keep track parameter values/reproductive number in each country/province
- Find the relationship of reproductive number and measures taken in each country/province

If you have ideas or need new functionalities, please join this project.
Any suggestions with [Github Issues](https://github.com/lisphilar/covid19-sir/issues/new/choose) are always welcomed. Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.

## Installation and dataset preparation
We have the following options to start analysis with CovsirPhy. Datasets are not included in this package, but we can prepare them with `DataLoader` class.

||Installation|Dataset preparation|
|:---|:---|:---|
|Standard users|pip/pipenv|Automated with `DataLoader` class|
|Developers|git-cloning|Automated with `DataLoader` class|
|Kagglers (local environment)|git-cloning|Kaggle API and Python script|
|Kagglers (Kaggle platform)|pip|Kaggle Datasets|

<strong>[Installation and dataset preparation](https://lisphilar.github.io/covid19-sir/INSTALLATION.html) explains how to install and prepare datasets for all users.</strong>

Covsirphy is available at [PyPI (The Python Package Index): covsirphy](https://pypi.org/project/covsirphy/) and supports Python 3.7 or newer versions.
```
pip install covsirphy --upgrade
```
Main datasets will be retrieved via [COVID-19 Data Hub](https://covid19datahub.io/https://covid19datahub.io/) and the citation is  
Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub", Working paper, doi: 10.13140/RG.2.2.11649.81763.


## Usage
Please read the following documents.

- [Usage (quickest version)](https://lisphilar.github.io/covid19-sir/usage_quickest.html)
- [Usage (quick version)](https://lisphilar.github.io/covid19-sir/usage_quick.html)
- Example codes in ["example" directory of this repository](https://github.com/lisphilar/covid19-sir/tree/master/example)
- [Kaggle: COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)

## License: Apache License 2.0
Please refer to [LICENSE](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE) file.

## Citation
CovsirPhy Development Team (2020), CovsirPhy, Python package for COVID-19 analysis with SIR-derived ODE models, https://github.com/lisphilar/covid19-sir

## Related work
Method of analysis in CovsirPhy:  
Lisphilar (2020), Kaggle notebook, COVID-19 data with SIR model, https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

Reproduction number evolution in each country:  
Ilyass Tabiai and Houda Kaddioui (2020), GitHub pages, COVID19 R0 tracker, https://ilylabs.github.io/projects/COVID-trackers/
