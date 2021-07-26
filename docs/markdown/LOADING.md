# Data loading (To Be Released)

From version 2.22.0, `DataLoader` will support to use local CSV files and `pandas.DataFrame`. (We can try this new feature with the latest development version.) Workflow of CovsirPhy analysis will be as follows.

Here, how to "prepare datasets" (the first step of CovsirPhy workflow) will be explained.

## 1-1. Decide whether to use the recommended datasets

As the first step, please create `DataLoader` instance. As default, the recommended datasets will be download and saved to "input" directory of the current directory. These downloaded datasets will be updated automatically when `DataLoader` instance is created and 12 hours passed since the last downloading.

```Python
import covsirphy as cs
loader = cs.DataLoader()
```

If you want to change the download directory (e.g. "datasets"), please use `directory` argument. the interval of downloading (default: 12 hours) can be changed with `update_interval` argument (e.g. 24 hours).

```Python
import covsirphy as cs
loader = cs.DataLoader(directory="datasets", update_interval=24)
```

If you want to use **ONLY** your own datasets, please set `update_interval=None`. `directory` argument will be ignored.

```Python
import covsirphy as cs
loader = cs.DataLoader(update_interval=None)
```

## 1-2. (Optional) Read datasets saved in our local environment

We can load our own datasets with `DataLoader.read_csv()` and `DataLoader.read_dataframe()`. When we use only recommended datasets, we can skip this step.

### 1-2-1. Variables to use

Variables to analyse are defined by `covsirphy`. (If you want to use a new dataset for your analysis, kindly create an issue with [GitHub Issues: Request new method of DataLoader class](https://github.com/lisphilar/covid19-sir/issues/new/?template=request-new-method-of-dataloader-class.md)! Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.)

Required:

- date: observation dates
- country: country names (the top level administration)
- province: province/state/prefecture names (the second level adminitration)
- confirmed: the number of confirmed cases
- fatal: the number of fatal cases
- population population values

Optional:

- iso3: ISO codes of countries
- recovered: the number of recovered cases
- tests: the number of tests
- product: vaccine product names
- vaccinations: cumulative number of vaccinations
- vaccinated_once: cumulative number of people who received at least one vaccine dose
- vaccinated_full: cumulative number of people who received all doses prescrived by the protocol

### 1-2-2. Read datasets from CSV files

To Be Edited (TBE), `DataLoader.read_csv()` and `DataLoader.assign()`. When we use only recommended datasets, we can skip this step.

### 1-2-3. Read datasets from pandas.DataFrame

TBE, `DataLoader.read_dataframe()`. When we use only recommended datasets, we can skip this step.

## 1-3. (Auto) Download the recommended datasets

If `update_interval` was not `None` when `DataLoader` instance was created, downloading of the recommended datasets will be done automatically. Downloading will be started with `DataLoader.lock()` or `DataLoader.jhu()` and so on, but they will be done at the next steps.

## 1-4. (Optional) Perform database lock

TBE, `DataLoader.lock()`.

## 1-5. (Auto) Clean data

TBE, `DataLoader.jhu()` etc.

```Python
# The number of cases and population values
jhu_data = loader.jhu()
# Government Response Tracker (OxCGRT)
oxcgrt_data = loader.oxcgrt()
# Population values
population_data = loader.population()
# The number of tests
pcr_data = loader.pcr()
# The number of vaccinations
vaccine_data = loader.vaccine()
# Population pyramid
pyramid_data = loader.pyramid()
# Japan-specific dataset
japan_data = loader.japan()
```

## Summary

TBE, workflow of methods (only with recommended datasets, both, only local)

## Data loading in Kaggle Notebook

We can use the recommended datasets in [Kaggle](https://www.kaggle.com/) Notebook. The datasets are saved in "/kaggle/input/" directory. Additionally, we can use Kaggle Datasets (CSV files) with `covsirphy` in Kaggle Notebook.

Note:  
If you have Kaggle API, you can download Kaggle datasets to your local environment by updating and executing [input.py](https://github.com/lisphilar/covid19-sir/blob/master/input.py) script. CSV files will be saved in "/kaggle/input/" directory.

Kaggle API:  
Move to account page of Kaggle and download "kaggle.json" by selecting "API > Create New API Token" button. Copy the json file to the top directory of the local repository or "~/.kaggle". Please refer to [How to Use Kaggle: Public API](https://www.kaggle.com/docs/api) and [stackoverflow: documentation for Kaggle API *within* python?](https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python#:~:text=Here%20are%20the%20steps%20involved%20in%20using%20the%20Kaggle%20API%20from%20Python.&text=Go%20to%20your%20Kaggle%20account,json%20will%20be%20downloaded)

## Acknowledgement

In Feb2020, CovsirPhy project started in Kaggle platform with [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) notebook using the following datasets.

- The number of cases (JHU) and linelist: [Novel Corona Virus 2019 Dataset by SRK](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
- Population in each country:  [covid19 global forecasting: locations population by Dmitry A. Grechka](https://www.kaggle.com/dgrechka/covid19-global-forecasting-locations-population)
- The number of cases in Japan: [COVID-19 dataset in Japan by Lisphilar](https://www.kaggle.com/lisphilar/covid19-dataset-in-japan)

Best Regards.
