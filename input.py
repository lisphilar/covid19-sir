#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path


def main():
    # Current directory
    cwd = Path.cwd()
    # The KAGGLE_CONFIG_DIR env var must be setup before loading KaggleApi
    if Path("kaggle.json").exists():
        os.environ["KAGGLE_CONFIG_DIR"] = cwd + "/"
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        # Put your kaggle.json in the directory which has input.py
        raise FileNotFoundError(
            f"Please save kaggle.json in {os.environ['KAGGLE_CONFIG_DIR']} or ~/.kaggle")
    api = KaggleApi()
    api.authenticate()
    # Remove the saved dataset
    top_path = Path(cwd) / "kaggle" / "input"
    top_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(top_path)
    # Download Kaggle datasets
    kaggle_datasets = [
        # author/title
        "sudalairajkumar/novel-corona-virus-2019-dataset",
        "dgrechka/covid19-global-forecasting-locations-population",
        "marcoferrante/covid19-prevention-in-italy",
        "hotessy/population-pyramid-2019",
        "lisphilar/covid19-dataset-in-japan",
    ]
    for name in kaggle_datasets:
        # Files will be saved in sub-directories
        dirpath = top_path / name.split("/")[1]
        api.dataset_download_files(name, path=dirpath, unzip=True)
    print("Datasets were successfully saved in kaggle/input directory.")


if __name__ == "__main__":
    main()
