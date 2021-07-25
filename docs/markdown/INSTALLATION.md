# Installation

`covsirphy` library supports Python 3.7 or newer versions.

**Please use `covsirphy` with a virtual environment** (venv/poetry/conda etc.) because it has many dependencies as listed in "tool.poetry.dependencies" of [pyproject.toml](https://github.com/lisphilar/covid19-sir/blob/master/pyproject.toml).

If you have any concerns, kindly create issues in [CovsirPhy: GitHub Issues page](https://github.com/lisphilar/covid19-sir/issues). All discussions are recorded there.

## Stable version

The latest stable version of CovsirPhy is available at [PyPI (The Python Package Index): covsirphy](https://pypi.org/project/covsirphy/).

```bash
pip install --upgrade covsirphy
```

Please check the version number as follows.

```Python
import covsirphy as cs
cs.__version__
```


## Development version

You can find the latest development in [GitHub repository: CovsirPhy](https://github.com/lisphilar/covid19-sir) and install it with `pip` command.

```bash
pip install --upgrade "git+https://github.com/lisphilar/covid19-sir.git#egg=covsirphy"
```

If you have a time to contribute CovsirPhy project, please refer to [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html). Always welcome!

## Installation with Anaconda

Anaconda users can install `covsirphy` in a conda environment (named "covid” for example). To avoid version conflicts of dependencies, `fiona`, `ruptures` and `pip` should be installed with conda command in advance.

```bash
conda create -n covid python=3 pip
conda activate covid
conda install -c conda-forge fiona ruptures
pip install --upgrade covsirphy
```

To exit this conda environment, please use `conda deactivate`.

# Dataset preparation

With `DataLoader` class, we can download recommended datasets for analysis and save/update them in your local environment. Optionally, you can use your local dataset which is saved in a CSV file.

All raw datasets are retrieved from public databases. No confidential information is included. If you find any issues, please let us know via [GitHub issue page](https://github.com/lisphilar/covid19-sir/issues).

## 1. Recommended datasets

With the following codes, we can download the latest recommended datasets and save them in "input" folder of the current directory. Please refer to [Usage (datasets)](https://lisphilar.github.io/covid19-sir/usage_dataset.html) to find the details of the datasets.

At first, import CovsirPhy package and check the version number.

```Python
import covsirphy as cs
cs.__version__
```

Save the recommended datasets in "input" folder of the current directory.  

```Python
# Create DataLoader instance
loader = cs.DataLoader("input")
# The number of cases and population values
jhu_data = loader.jhu()
# Government Response Tracker (OxCGRT)
oxcgrt_data = loader.oxcgrt()
```

```Python
# Population values
population_data = loader.population()
# Linelist of case reports
linelist = loader.linelist()
# The number of tests
pcr_data = loader.pcr()
# The number of vaccinations
vaccine_data = loader.vaccine()
# Population pyramid
pyramid_data = loader.pyramid()
# Japan-specific dataset
japan_data = loader.japan()
```

The downloaded datasets were retrieved from the following sites.

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

## 2. How to request new data loader

If you want to use a new dataset for your analysis, please kindly inform us using [GitHub Issues: Request new method of DataLoader class](https://github.com/lisphilar/covid19-sir/issues/new/?template=request-new-method-of-dataloader-class.md). Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.

## 3. Use a local CSV file which has the number of cases

We can replace `jhu_data` instance created by `DataLoader` class with your dataset saved in a CSV file. At this time, `covsirphy` supports country and province level data.

### 3.1. Create CountryData instance

Please create `CountryData` instance at first. Let's say we have a CSV file ("oslo.csv") with the following columns.

- "date": reported dates
- "confirmed": the number of confirmed cases
- "recovered": the number of recovered cases
- "fatal": the number of fatal cases
- "province": (optional) province names

Country level data will be set as total values of provinces with `CountryData.register_total()` method optionally.

```Python
# Create CountryData instance specifying filename and country name
country_data = cs.CountryData("oslo.csv", country="Norway")
# Specify column names
country_data.set_variables(
    date="date", confirmed="confirmed", recovered="recovered", fatal="fatal", province="province",
)
# (Optional) register total values of provinces as country level data
country_data.register_total()
# Check records -> pandas.DataFrame
# reset index, Date/Country/Province/Confirmed/Infected/Fatal/Recovered column
country_data.cleaned()
```

When we don't have province column and the all records are for one province, we can specify the province name as follows.

```Python
# Create CountryData instance specifying filename and country/province name
country_data = cs.CountryData("oslo.csv", country="Norway", province="Oslo")
# Specify column names except for province
country_data.set_variables(
    date="date", confirmed="confirmed", recovered="recovered", fatal="fatal",
)
# Check records
country_data.cleaned()
```

When we don't have province column and the all records are country level data, we can skip province name setting.

```Python
# Create CountryData instance specifying filename and country name
country_data = cs.CountryData("oslo.csv", country="Norway")
# Specify column names except for province
country_data.set_variables(
    date="date", confirmed="confirmed", recovered="recovered", fatal="fatal",
)
# Check records
country_data.cleaned()
```

If dates are parsed incorrectly (e.g. the last date of raw dataset was 12Jun2021, but that of cleaned dataset was 06Dec2021), please try the following codes with appropreate date format for a while.

```Python
import pandas as pd
# Remove cleaned data with wrong time format
country_data._cleaned_df = pd.DataFrame()
# Update raw dataframe with appropreate time format
country_data.raw["Date"] = pd.to_datetime(country_data.raw["Date"], format="%d/%m/%Y")
# Data cleaning
country_data.cleaned()
```

From development version 2.21.0-delta, we can use the following. This will be implemented at the next stable version 2.22.0.

```Python
country_data.cleaned(date_format="%d/%m/%Y")
```


### 3.2. Convert to JHUData instance

Then, convert the `CountryData` instance to a `JHUData` instance.

```Python
# Create JHUData instance using cleaned dataset (pandas.DataFrame)
jhu_data = cs.JHUData.from_dataframe(country_data.cleaned())
# Or, we can use and update the output of DataLoader.jhu()
# jhu_data = data_loader.jhu()
# jhu_data.replace(country_data)
```

### 3.3. Set population values

Additionally, you may need to register population values to `PopulationData` instance manually.

```Python
# Create PopulationData instance with empty dataset
population_data = cs.PopulationData()
# Or, we can use the output of DataLoader.population()
# population_data = data_loader.population()
# Update the population value: province is optional
population_data.update(693494, country="Norway", province="Oslo")
```


# Data loading (To Be Released)

From version 2.22.0, `DataLoader` will support to use local CSV files and `pandas.DataFrame`. (We can try this new feature with the latest development version.) Workflow of CovsirPhy analysis will be as follows.

1. Prepare datasets
    1. Decide whether to use the recommended datasets
    2. (Optional) Read datasets saved in our local environment
    3. (Auto) Download the recommended datasets
    4. (Optional) Perform database lock
    5. (Auto) Clean data
2. [Perform Exploratory data analysis](https://lisphilar.github.io/covid19-sir/usage_dataset.html)
3. [Learn SIR-derived models](https://lisphilar.github.io/covid19-sir/usage_theoretical.html)
4. [Learn S-R trend analysis](https://lisphilar.github.io/covid19-sir/usage_phases.html)
5. [Perform scenario analysis](https://lisphilar.github.io/covid19-sir/usage_quick.html)
    1. Register cleaned data
    2. Check records of the selected country/province
    3. Perform S-R trend analysis to split time series data to phases
    4. Estimate ODE parameter values in the past phases
    5. (Experimental) Predict ODE parameter values in the future phases
    6. Simulate the number of cases with some scenarios
    7. (To Be Implemented) Find solutions to end the outbreak

Here, how to "prepare datasets" (the first step) will be explained.

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
