#!/bin/bash
# https://www.kaggle.com/docs/api
# Using Kaggle API, download the datasets

# Initialize the input directory
mkdir -p input
rm input/*.csv

# Download datasets

# The number of cases
kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset
# The number of cases in Japan
kaggle datasets download -d lisphilar/covid19-dataset-in-japan
# Total population
kaggle datasets download -d dgrechka/covid19-global-forecasting-locations-population

# Move files to input direcotory
mv *.zip input/

# Unzip the files
unzip 'input/*.zip' -d input
rm input/*.zip

# Remove un-used CSV files
rm input/*time_series_covid_19_*.csv
rm input/COVID19_line_list_data.csv
rm input/COVID19_open_line_list.csv
