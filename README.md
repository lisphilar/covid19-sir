# COVID-19 data with SIR model [![GitHub license](https://img.shields.io/github/license/lisphilar/covid19-sir)](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE.md)[![Python version](https://img.shields.io/badge/Python-3.7|3.8-green.svg)](https://www.python.org/)
This is a package for COVID-19 data analysis with SIR-derived models. Please refer to [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) notebook in Kaggle to understand the methods of analysis.

## Introduction
With this Python package we can apply SIR-F model to COVID-19 data. SIR-F is a customized ODE model derived from SIR model. To evaluate the effect of measures, parameter estimation of SIR-F will be applied to subsets of time series data in each country. Parameter change points will be determined by S-R trend analysis. The details are explained in [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) in Kaggle.

## Recomended datasets
The datasets can be download using Kaggle API key and Kaggle package. Please read [How to Use Kaggle: Public API](https://www.kaggle.com/docs/api) and my Bash code `input.sh` in this repository.
### The number of cases
Primary source: [COVID-19 Data Repository by CSSE at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)  
Secondary source: [Novel Corona Virus 2019 Dataset by SRK](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)  
### Total population
[covid19 global forecasting: locations population by Dmitry A. Grechka](https://www.kaggle.com/dgrechka/covid19-global-forecasting-locations-population)  
### The number of cases in Japan
Primary source: [Ministry of Health, Labour and Welefare HP (in English)](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/newpage_00032.html)  
Secondary source: [Secondary source: COVID-19 dataset in Japan by Lisphilar](https://www.kaggle.com/lisphilar/covid19-dataset-in-japan)  


## Installation
Please install this package using
- pipenv install ```pipenv install git+https://github.com/lisphilar/covid19-sir#egg=covsirphy```, or
- pip install ```pip install git+https://github.com/lisphilar/covid19-sir#egg=covsirphy```, or
- Clone this repository ```git clone https://github.com/lisphilar/covid19-sir.git```

## Usage
(Updated the codes will be uploaded)

## Citation
Lisphilar, 2020, Kaggle notebook, COVID-19 data with SIR model, https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

Lisphilar, 2020, GitHub repository, https://github.com/lisphilar/covid19-sir
