# CovsirPhy: COVID-19 with SIRs
[![PyPI version](https://badge.fury.io/py/covsirphy.svg)](https://badge.fury.io/py/covsirphy)
[![Downloads](https://pepy.tech/badge/covsirphy)](https://pepy.tech/project/covsirphy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/covsirphy)](https://badge.fury.io/py/covsirphy)  
[![GitHub license](https://img.shields.io/github/license/lisphilar/covid19-sir)](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE)
[![Maintainability](https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/maintainability)](https://codeclimate.com/github/lisphilar/covid19-sir/maintainability)
[![test](https://github.com/lisphilar/covid19-sir/workflows/test/badge.svg)](https://github.com/lisphilar/covid19-sir/actions)

<strong>CovsirPhy is a Python package for COVID-19 (Coronavirus disease 2019) data analysis with SIR-derived ODE models. Please refer to "Method" part of [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) notebook in Kaggle to understand the methods.</strong>

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

### Standard users
Covsirphy is available at [PyPI (The Python Package Index): covsirphy](https://pypi.org/project/covsirphy/) and supports Python 3.7 or newer versions.
```
pip install covsirphy
```

Then, download the datasets with the following codes, when you want to save the data in `input` directory.
```Python
import covsirphy as cs
data_loader = cs.DataLoader("input")
jhu_data = data_loader.jhu()
japan_data = data_loader.japan()
population_data = data_loader.population()
oxcgrt_data = data_loader.oxcgrt()
```
If `input` directory has the datasets, `DataLoader` will load the local files. If the datasets were updated in remote servers, `DataLoader` will update the local files automatically.

We can get descriptions of the datasets and raw/cleaned datasets easily. As an example, JHU dataset will be used here.
```Python
# Description (string)
jhu_data.citation
# Raw data (pandas.DataFrame)
jhu_data.raw
# Cleaned data (pandas.DataFrame)
jhu_data.cleaned()
```
We can get COVID-19 Data Hub citation list of primary sources as follows.
```Python
data_loader.covid19dh_citation
```


# Quick usage for analysis
Example Python codes are in `example` directory. With Pipenv environment, we can run the Python codes with Bash code `example.sh` in the top directory of this repository.

## Preparation
```Python
import covsirphy as cs
cs.__version__
```
Please load the datasets as explained in the previous section.

## Scenario analysis
As an example, use dataset in Italy.

### Check records
```Python
ita_scenario = cs.Scenario(jhu_data, population_data, country="Italy", province=None)
```
See the records as a figure.
```Python
ita_record_df = ita_scenario.records()
```
### S-R trend analysis
Perform S-R trend analysis and set phases to the scenario.
The number of change points will be determined automatically (>= 2.4.0).
```Python
ita_scenario.trend(set_phases=True)
print(ita_scenario.summary())
```
### Hyperparameter estimation of ODE models
As an example, use SIR-F model.
```Python
ita_scenario.estimate(cs.SIRF)
print(ita_scenario.summary())
```
We can check the accuracy of estimation with a figure.
```Python
# Table
ita_scenario.estimate_accuracy(phase="1st")
# Get a value
ita_scenario.get("Rt", phase="4th")
# Show parameter history as a figure
ita_scenario.param_history(targets=["Rt"], divide_by_first=False, box_plot=False)
ita_scenario.param_history(targets=["rho", "sigma"])
```
### Prediction of the number of cases
we can add some future phases.
```Python
# if needed, clear the registered future phases
ita_scenario.clear(name="Main")
# Add future phase to main scenario
ita_scenario.add_phase(name="Main", end_date="01Aug2020")
# Get parameter value
sigma_4th = ita_scenario.get("sigma", name="Main", phase="4th")
# Add future phase with changed parameter value to new scenario
sigma_6th = sigma_4th * 2
ita_scenario.add_phase(end_date="31Dec2020", name="Medicine", sigma=sigma_6th)
ita_scenario.add_phase(days=30, name="Medicine")
print(ita_scenario.summary())
```
Then, we can predict the number of cases and get a figure.
```Python
# Prediction and show figure
sim_df = ita_scenario.simulate(name="Main")
# Describe representative values
print(ita_scenario.describe())
```

# Information

## Apache License 2.0
Please refer to [LICENSE](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE) file.

## Citation
CovsirPhy Development Team (2020), CovsirPhy, Python package for COVID-19 analysis with SIR-derived ODE models, https://github.com/lisphilar/covid19-sir

## Related work
Method of analysis in CovsirPhy:  
Lisphilar (2020), Kaggle notebook, COVID-19 data with SIR model, https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

Reproduction number evolution in each country:  
Ilyass Tabiai and Houda Kaddioui (2020), GitHub pages, COVID19 R0 tracker, https://ilylabs.github.io/projects/COVID-trackers/
