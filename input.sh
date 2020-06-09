#!/bin/bash
# https://www.kaggle.com/docs/api
# Using Kaggle API, download the datasets

# Initialize the input directory
mkdir -p input
rm input/*.csv 2>/dev/null

# Download datasets from Kaggle

# The number of cases
pipenv run kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset
# The number of cases in Japan
pipenv run kaggle datasets download -d lisphilar/covid19-dataset-in-japan
# Total population
pipenv run kaggle datasets download -d dgrechka/covid19-global-forecasting-locations-population

# Move files to input direcotory
mv *.zip input/

# Unzip the files
unzip 'input/*.zip' -d input
rm input/*.zip

# Remove un-used CSV files
rm input/*time_series_covid_19_*.csv
rm input/COVID19_line_list_data.csv
rm input/COVID19_open_line_list.csv


# Download datasets from GitHub
# sudo apt install subversion

# Oxford Covid-19 Government Response Tracker (OxCGRT)
svn export https://github.com/OxCGRT/covid-policy-tracker/trunk/data input/oxcgrt --force --non-recursive
