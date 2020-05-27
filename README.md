# COVID-19 data with SIR model [![GitHub license](https://img.shields.io/github/license/lisphilar/covid19-sir)](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE.md)[![Python version](https://img.shields.io/badge/Python-3.7|3.8-green.svg)](https://www.python.org/)
This is a package for COVID-19 data analysis with SIR-derived models. Please refer to [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) notebook in Kaggle to understand the methods of analysis.

## Introduction
With this Python package we can apply SIR-F model to COVID-19 data. SIR-F is a customized ODE model derived from SIR model. To evaluate the effect of measures, parameter estimation of SIR-F will be applied to subsets of time series data in each country. Parameter change points will be determined by S-R trend analysis. The details are explained in [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) in Kaggle.

## Recomended datasets
The datasets can be download using Kaggle API key and Kaggle package. Please read [How to Use Kaggle: Public API](https://www.kaggle.com/docs/api) and my Bash code `input.sh` in this repository.
### The number of cases
Primary source: [COVID-19 Data Repository by CSSE at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)  
Secondary source: [Novel Corona Virus 2019 Dataset by SRK](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)  
### The number of cases in Japan
Primary source: [Ministry of Health, Labour and Welefare HP (in English)](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/newpage_00032.html)  
Secondary source: [Secondary source: COVID-19 dataset in Japan by Lisphilar](https://www.kaggle.com/lisphilar/covid19-dataset-in-japan)  
### Total population
[covid19 global forecasting: locations population by Dmitry A. Grechka](https://www.kaggle.com/dgrechka/covid19-global-forecasting-locations-population)  


## Installation
When you use this package in Kaggle notebook (need to turn on Internet option in notebook settings) or local environment with Pip,
```
pip install git+https://github.com/lisphilar/covid19-sir#egg=covsirphy
```
With Pipenv environment,
```
pipenv install git+https://github.com/lisphilar/covid19-sir#egg=covsirphy
```
For developers,
```
git clone https://github.com/lisphilar/covid19-sir.git
pipenv install --dev
```

## Quick usage
### Preparation
```Python
import covsirphy as cs
```
Perform data cleaning of JHU dataset.
```Python
# With CSV filepath of JHU dataset
jhu_data = cs.JHUData("input/covid_19_data.csv")
ncov_df = jhu_data.cleaned()
```
(Optional) We can replace JHU data with country-specific dataset.
```Python
# As an example, read Japan dataset
jpn_data = cs.CountryData("input/covid_jpn_total.csv", country="Japan")
jpn_data.set_variables(
    # Specify the column names to read
    date="Date",
    confirmed="Positive",
    fatal="Fatal",
    recovered="Discharged",
    province=None
)
# Replacement of JHU data in Japan
jhu_data = jhu_data.replace(jpn_data)
ncov_df = jhu_data.cleaned()
```
Perform data cleaning of population dataset.
```Python
# With CSV filepath of population dataset
pop_data = cs.Population("input/locations_population.csv")
# We can add a new record
pop_data.update(country="Example", province="-", value=1_000_000)
# Return dictionary
pop_data.to_dict(country_level=True)
```
### Scenario analysis
As an example, use dataset in Italy.
#### Check records
```Python
ita_scenario = cs.Scenario(jhu_data, pop_data, country="Italy", province=None)
```
See the records as a figure.
```Python
ita_record_df = ita_scenario.records()
#### S-R trend analysis
```
Show S-R trend and determine the number of change points.
```Python
ita_scenario.trend()
```
As an example, set the number of change points as 4.
```Python
ita_scenario.trend(n_points=4, set_phase=True)
```
Start/end date of the four phase were automatically determined. Let's see.
```Python
print(ita_scenario.summary())
```
#### Hyperparameter estimation of ODE models
As an example, use SIR-F model.
```Python
ita_scenario.estimate(cs.SIRF)
print(ita_scenario.summary())
```
We can check the accuracy of estimation with a figure.
```Python
ita_scenario.estimate_accuracy(phase="1st")
```

(Update later)

## Citation
Lisphilar, 2020, Kaggle notebook, COVID-19 data with SIR model, https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

Lisphilar, 2020, GitHub repository, Covsirphy, Python package for COVID-19 data with SIR model, https://github.com/lisphilar/covid19-sir
