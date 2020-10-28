#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

# Remove the saved dataset
path_ = cwd + "/kaggle/input/"
os.makedirs(path_, exist_ok=True)
shutil.rmtree(path_)

# Download Kaggle datasets
kaggle_datasets = [
    "sudalairajkumar/novel-corona-virus-2019-dataset",
    "dgrechka/covid19-global-forecasting-locations-population",
    "marcoferrante/covid19-prevention-in-italy",
    "lisphilar/covid19-dataset-in-japan",
]

for dataset in kaggle_datasets:
    api.dataset_download_files(dataset, path=path_, unzip=True)

# Download Kaggle population pyramid datasets
# Extract the datasets in population-pyramid-2019 folder
population_pyramid_path_ = path_ + "/population-pyramid-2019"
kaggle_dataset_population_pyramid = "hotessy/population-pyramid-2019"
api.dataset_download_files(kaggle_dataset_population_pyramid, path=population_pyramid_path_, unzip=True)

# OxCGRT
oxcgrt_file = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

r = requests.get(oxcgrt_file, allow_redirects=True)
with open(path_ + oxcgrt_file.rsplit('/', 1)[-1], 'wb') as fh:
    fh.write(r.content)
