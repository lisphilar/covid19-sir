#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
from pathlib import Path
import requests

# The KAGGLE_CONFIG_DIR env var must be setup before loading KaggleApi
cwd = os.getcwd()
if Path("./kaggle.json").exists():
    os.environ["KAGGLE_CONFIG_DIR"] = cwd + "/"

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    # Put your kaggle.json in the same folder as input.py
    raise FileNotFoundError(
        f"Please save kaggle.json in {os.environ['KAGGLE_CONFIG_DIR']} or ~/.kaggle")

api = KaggleApi()
api.authenticate()
path_ = cwd + "/input/"
os.makedirs(path_, exist_ok=True)

shutil.rmtree(path_)

api.dataset_download_files('dgrechka/covid19-global-forecasting-locations-population',
                           path=path_ + "/",
                           unzip=True)

api.dataset_download_files('sudalairajkumar/novel-corona-virus-2019-dataset',
                           path=path_ + "/",
                           unzip=True)

api.dataset_download_files('lisphilar/covid19-dataset-in-japan',
                           path=path_ + "/",
                           unzip=True)

# Removing files that will not be neccessary
file_list = glob.glob(path_ + '/*')
file_list_keep = file_list
file_list_keep = [
    ele for ele in file_list_keep if "time_series_covid_19_" not in ele]
file_list_keep = [
    ele for ele in file_list_keep if "COVID19_line_list_data.csv" not in ele]
file_list_keep = [
    ele for ele in file_list_keep if "COVID19_open_line_list.csv" not in ele]
file_list_keep = [
    ele for ele in file_list_keep if "covid_jpn_metadata.csv" not in ele]
file_list_keep = [
    ele for ele in file_list_keep if "covid_jpn_prefecture.csv" not in ele]

for file_ in file_list:
    if file_ not in file_list_keep:
        os.remove(file_)

oxcgrt_file = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

r = requests.get(oxcgrt_file, allow_redirects=True)
with open(path_ + oxcgrt_file.rsplit('/', 1)[-1], 'wb') as fh:
    fh.write(r.content)
