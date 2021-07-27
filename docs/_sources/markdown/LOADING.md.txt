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

The datasets downloaded automatically downloaded is listed here. Please refer to [Usage (datasets)](https://lisphilar.github.io/covid19-sir/usage_dataset.html) to find the details of the datasets.

### [COVID-19 Data Hub](https://covid19datahub.io/)

Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub", Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.

- The number of cases (JHU style)
- Population values in each country/province
- [Government Response Tracker (OxCGRT)](https://github.com/OxCGRT/covid-policy-tracker)
- The number of tests

### [Our World In Data](https://github.com/owid/covid-19-data/tree/master/public/data)

Hasell, J., Mathieu, E., Beltekian, D. et al. A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020). [https://doi.org/10.1038/s41597-020-00688-8](https://doi.org/10.1038/s41597-020-00688-8)

- The number of tests
- The number of vaccinations
- The number of people who received vaccinations

### [COVID-19 Open Data by Google Cloud Platform](https://github.com/GoogleCloudPlatform/covid-19-open-data)

O. Wahltinez and others (2020), COVID-19 Open-Data: curating a fine-grained, global-scale data repository for SARS-CoV-2, Work in progress, [https://goo.gle/covid-19-open-data](https://goo.gle/covid-19-open-data)

- percentage to baseline in visits (will be usable from 2.22.0)

Note:  
**Please refer to [Google Terms of Service](https://policies.google.com/terms) in advance.**

### [World Bank Open Data](https://data.worldbank.org/)

World Bank Group (2020), World Bank Open Data, [https://data.worldbank.org/](https://data.worldbank.org/)

- Population pyramid

### [Datasets for CovsirPhy](https://github.com/lisphilar/covid19-sir/tree/master/data)

Hirokazu Takaya (2020-2021), GitHub repository, COVID-19 dataset in Japan, [https://github.com/lisphilar/covid19-sir/tree/master/data](https://github.com/lisphilar/covid19-sir/tree/master/data).  

- The number of cases in Japan (total/prefectures)
- Metadata regarding Japan prefectures

### How to request new data loader

If you want to use a new dataset for your analysis, please kindly inform us using [GitHub Issues: Request new method of DataLoader class](https://github.com/lisphilar/covid19-sir/issues/new/?template=request-new-method-of-dataloader-class.md). Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.

## 1-4. Perform database lock

`DataLoader.lock()` (method for database) is required when you want to use local CSV files and `pandas.DataFrame` as the database. (Please skip this step if you use ONLY the automatically-downloaded datasets.)

To use the local datasets, we need to link the column names of the local database to the variable names defined by CovsirPhy project. This can be done as follows.
As an example, we assume that all variables are registered by `DataLoader.read_csv()`, `DataLoader.read_dataframe()` and `DataLoader.assign()`.

```Python
loader.lock(
    # Always required
    date="date", country="country", province="province",
    confirmed="confirmed", fatal="fatal", population="population",
    # Optional regarding location
    iso3="iso3",
    # Optional regarding JHUData
    recovered="recovered",
    # Optional regarding PCData
    tests="tests",
    # Optional regarding VaccineData
    product="product", vaccinations="vaccinations",
    vaccinated_once="vaccinated_once", vaccinated_full="vaccinated_full",
    # Optinal for OxCGRTData (list[str])
    oxcgrt_variables=None,
    # Optinal for OxCGRTData (list[str])
    mobility_variables=None
)

```

## 1-5. Clean data

TBE, `DataLoader.jhu()` etc.

```Python
# The number of cases and population values
jhu_data = loader.jhu()
# Government Response Tracker (OxCGRT)
oxcgrt_data = loader.oxcgrt()
# The number of tests
pcr_data = loader.pcr()
# The number of vaccinations
vaccine_data = loader.vaccine()
# Mobility data (will be impremented, from version 2.22.0)
# mobility_data = loader.mobility()
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
