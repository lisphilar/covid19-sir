# Usage: data loading

The first step of CovsirPhy analysis is data laoding. In this chapter, we will confirm the way to load the following two type of datasets using `DataLoader` class.

- recommented datasets: open datasets recommended by CovsirPhy community
- local datasets: opened/closed datasets you have in your local environment

## 1. Decide whether to use the recommended datasets

We will create `DataLoader` instance. As default, the recommended datasets will be download and saved to "input" directory of the current directory. These downloaded datasets will be updated automatically when `DataLoader` instance is created and 12 hours have passed since the last downloading.

```Python
import covsirphy as cs
loader = cs.DataLoader()
```

If you want to change the download directory (e.g. "datasets"), please use `directory` argument. the interval of downloading (default: 12 hours) can be changed with `update_interval` argument (e.g. 24 hours).

```Python
import covsirphy as cs
loader = cs.DataLoader(directory="datasets", update_interval=24)
```

If you want to use **ONLY** local datasets, please set `update_interval=None`. `directory` argument will be ignored.

```Python
import covsirphy as cs
loader = cs.DataLoader(update_interval=None)
```

## 2. Read local datasets

We can read local datasets with `DataLoader.read_csv()`, `DataLoader.read_dataframe()` and `DataLoader.assign()`. When we use only recommended datasets, we can skip this step.

### 2-1. Check variables to use

Variables to analyse are specified by `covsirphy`. Please check that you have records of the required variables and the correspondence of te variables you have and variables specified by `covsirphy`. The required variables must be prepared with `DataLoader.read_csv()`, `DataLoader.read_dataframe()` and `DataLoader.assign()`. We can decide column names freely at this step and we will tell the correspondence to `DataLoader` with `DataLoader.lock()` later.

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

Changeable:

- variables of government response indicators
- variables of mobility indicators

Note that `covsirphy` uses only two levels of administration (country and province). If you have the third level (e.g. city), please regard it as province for analysis.

If you want to use a new dataset for your analysis, kindly create an issue with [GitHub Issues: Request new method of DataLoader class](https://github.com/lisphilar/covid19-sir/issues/new/?template=request-new-method-of-dataloader-class.md)! Please read [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html) in advance.

### 2-2. Read datasets from CSV files

If we have records as CSV files (time series data of vairables), we can read them with `DataLoader.read_csv()` method. This uses `pandas.read_csv()` internally and [arguments](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) of `pandas.read_csv()` can be used.

As an example, we have records in "./usa.csv" as shown in the next table. Obseervation dates are assigned to "date" column with "YYYY-MM-DD" format. (Data is from [COVID-19 Data Hub](https://covid19datahub.io/).)

|    | confirmed | fatal | province | population | date       |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0  | 966287    | 17979 | Illinois | 12671821   | 2021-01-01 |
| 1  | 977677    | 18217 | Illinois | 12671821   | 2021-01-02 |
| 2  | 982105    | 18322 | Illinois | 12671821   | 2021-01-03 |

We can read it with `DataLoader.read_csv()` method. Argument `parse_dates` (list of columns of dates) and `dayfirst` (whether date and month are formatted with DD/MM or not) are optional, but it is suggested to use them to read date information correctly. The read data is saved as a `pandas.DataFrame` and it can be checked with `DataLoader.local` property.

```Python
loader.read_csv("./usa.csv", parse_dates=["date"], dayfirst=False)
print(loader.local)
```

If you have multiple CSV files, you can call `DataLoader.read_csv()` multiple times. Note that we need to specify how to combine the current data and the new data with `how_combine` (string) argument. Candidates of `how_combine` is listed here.

- "replace" (default): replace registered dataset with the new data
- "concat": concat datasets with `pandas.concat()`
- "merge": merge datasets with `pandas.DataFrame.merge()`
- "update": update the current dataset with `pandas.DataFrame.update()`

```Python
loader.read_csv("./usa.csv", parse_dates=["date"], dayfirst=False)
loader.read_csv("./uk.csv", parse_dates=["date"], dayfirst=True, how_combine="concat")
print(loader.local)
```

Because `DataLoader.read_csv()` uses `pandas.read_csv()` internally, URLs can be used as the first positional argument.

```Python
loader.read_csv(
    "https://github.com/lisphilar/covid19-sir/tree/master/data",
    parse_dates=["date"], dayfirst=False
)
print(loader.local)
```

### 2-3. Read datasets from pandas.DataFrame

If you have local datasets as a `pandas.DataFrame`, please use `DataLoader.read_dataframe()`. Its usage is similar to `DataLoader.read_csv()`. As an example, we the dataset as `usa_df` and `uk_df` (instance of `pandas.DataFrame`).

```Python
loader.read_csv(usa_df, parse_dates=["date"], dayfirst=False)
loader.read_csv(uk_df, parse_dates=["date"], dayfirst=True, how_combine="concat")
print(loader.local)
```

### 2-4. Assign columns

We can set variables using `DataLoader.assign()`. This use `pandas.DataFrame.assign()` internally and we can assign new variables (columns) with stable values and `lambda` function.

Let's say, we have the following dataset as `loader.local`. We want to assign

- country name (string "USA"),
- population values (12,671,821 persons), and
- the number of vaccinations as the total value of vaccinated_once and vaccinated_full.

(The values of vaccinated_once and vaccinated_full are not actual values. They are just simplified example values.)

|    | confirmed | fatal | province | date       | vaccinated_once | vaccinated_full |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0  | 966287    | 17979 | Illinois | 2021-01-01 | 1000            | 500             |
| 1  | 977677    | 18217 | Illinois | 2021-01-02 | 2000            | 700             |
| 2  | 982105    | 18322 | Illinois | 2021-01-03 | 3000            | 800             |

We can assign them as follows.

```Python
loader.assign(
    country="USA",
    population=12_671_821,
    vaccinations=lambda x: x["vaccinated_once"] + x["vaccinated_full"]
)
print(loader.local)
```

Three columns will be added.

|| confirmed | fatal | province | date | vaccinated_once | vaccinated_full | country | population | vaccinations |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 966287 | 17979 | Illinois | 2021-01-01 | 1000 | 500 | USA | 12671821 | 1500 |
| 1 | 977677 | 18217 | Illinois | 2021-01-02 | 2000 | 700 | USA | 12671821 | 2700 |
| 2  | 982105 | 18322 | Illinois | 2021-01-03 | 3000 | 800 | USA  | 12671821 | 3800 |

## 3. Perform database lock

We need to run `DataLoader.lock()` (method for database lock) when you want to use local CSV files and `pandas.DataFrame` as the database. (i.e. We can skip this method when you use **ONLY** the recommended datasets.) After completion of database lock, we cannot update local database with `DataLoader.read_csv()` and so on.

By database lock, we tell the correspondence of te variables you have and variables specified by `covsirphy` and lock the local database. Addtionally, the all recommended datasets will be downloaded automatically (if `update_interval` was not `None`) and combined to the local database.

Database lock can be done as follows. As an example, we assume that all variables are registered in advance.

- Argument names of `DataLoader.lock()` is listed at [2-1. Variables to use](https://lisphilar.github.io/covid19-sir/markdown/LOADING.html#variables-to-use).
- `oxcgrt_variables` (e.g. `["Stringency_index", "Contact_tracing"]`) is a variable name list for `OxCGRTData` (government response indicators).
- `mobility_variables` (e.g. `["Mobility_workplaces", "Mobility_residential"]`) is a variable name list for `MobilityData` (mobility indicators).

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
    # Optinal for OxCGRTData (list[str] or None)
    oxcgrt_variables=["Stringency_index", "Contact_tracing"],
    # Optinal for OxCGRTData (list[str] or None)
    mobility_variables=["Mobility_workplaces", "Mobility_residential"],
)
```

If you do not have some variables in the local database, please skip the arguments or apply `None` to the arguments. For example, the codes will be as follows if we have only the required arguemnts listed at [2-1. Variables to use](https://lisphilar.github.io/covid19-sir/markdown/LOADING.html#variables-to-use).

```Python
loader.lock(
    date="date", country="country", province="province",
    confirmed="confirmed", fatal="fatal", population="population",
)
```

`DataLoader.locked` is a read-only property to check the locked database. instance of `pandas.DataFrame` will be returned.

```Python
print(loader.locked)
```

## 4. Download the recommended datasets

If `update_interval` was not `None` when `DataLoader` instance was created, downloading of the recommended datasets will be started automatically with calling `DataLoader.lock()` or `DataLoader.jhu()` etc.

The recommended datasets are listed here. Please refer to [Usage (datasets)](https://lisphilar.github.io/covid19-sir/usage_dataset.html) to find the details of the datasets. All recommended datasets are retrieved from public databases. No confidential information is included. If you found any issues, please let us know via [GitHub issue page](https://github.com/lisphilar/covid19-sir/issues).

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

## 5. Clean data

`DataLoader.jhu()` and some methods listed here will create `JHUData` instance and so on respectively and performs data cleaning automatically.

```Python
# The number of cases and population values
jhu_data = loader.jhu()
# Government Response Tracker (OxCGRT)
oxcgrt_data = loader.oxcgrt()
# The number of tests
pcr_data = loader.pcr()
# The number of vaccinations
vaccine_data = loader.vaccine()
# Mobility data
mobility_data = loader.mobility()
# Population pyramid
pyramid_data = loader.pyramid()
# Japan-specific dataset
japan_data = loader.japan()
```

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
