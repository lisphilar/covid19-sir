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
