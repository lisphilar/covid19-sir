Installation
============

The latest stable version of CovsirPhy is available at `PyPI (The Python
Package Index): covsirphy <https://pypi.org/project/covsirphy/>`__ and
supports Python 3.6.9 or newer versions. It is recommended to use
virtual environment.

::

    pip install --upgrade covsirphy

The latest development version can be install from `GitHub repository:
CovsirPhy <https://github.com/lisphilar/covid19-sir>`__. Please refer to
`Guideline of
contribution <https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html>`__.

::

    pip install --upgrade "git+https://github.com/lisphilar/covid19-sir.git#egg=covsirphy"

| Note:
| When using **development versions** of CovsirPhy in Kaggle Notebook,
  please run the following codes to remove third-party ``typing``
  package.

.. code:: python

    !pip uninstall typing -y
    !pip install --upgrade "git+https://github.com/lisphilar/covid19-sir.git#egg=covsirphy"

Dataset preparation
===================

Recommended datasets for analysis can be downloaded and updated easily
with ``DataLoader`` class. If you have CSV files in your environment,
you can analyse them.

All raw datasets are retrieved from public databases. No confidential
information is included. If you find any issues, please let us know via
GitHub issue page.

1. Recommended datasets
-----------------------

With the following codes, we can download the latest recommended
datasets and save them in "input" folder of the current directory.
Please refer to `Usage
(datasets) <https://lisphilar.github.io/covid19-sir/usage_dataset.html>`__
to find the details of the datasets.

.. code:: python

    import covsirphy as cs
    cs.__version__
    # Create DataLoader instance
    data_loader = cs.DataLoader("input")

.. code:: python

    # (Main) The number of cases (JHU style)
    jhu_data = data_loader.jhu()
    # (Main) Population in each country
    population_data = data_loader.population()
    # (Main) Government Response Tracker (OxCGRT)
    oxcgrt_data = data_loader.oxcgrt()
    # Linelist of case reports
    linelist = data_loader.linelist()
    # The number of tests
    pcr_data = data_loader.pcr()
    # The number of vaccinations
    vaccine_data = data_loader.vaccine()
    # Population pyramid
    pyramid_data = data_loader.pyramid()
    # Japan-specific dataset
    japan_data = data_loader.japan()

The downloaded datasets were retrieved from the following sites.

`COVID-19 Data Hub <https://covid19datahub.io/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub", Journal of Open
Source Software 5(51):2376, doi: 10.21105/joss.02376.

-  The number of cases (JHU style)
-  Population in each country
-  `Government Response Tracker
   (OxCGRT) <https://github.com/OxCGRT/covid-policy-tracker>`__
-  The number of tests

`Open COVID-19 Data Working Group <https://github.com/beoutbreakprepared/nCoV2019>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xu, B., Gutierrez, B., Mekaru, S. et al. Epidemiological data from the
COVID-19 outbreak, real-time case information. Sci Data 7, 106 (2020).
https://doi.org/10.1038/s41597-020-0448-0

-  Linelist of case reports

`Our World In Data <https://github.com/owid/covid-19-data/tree/master/public/data>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Citation: Hasell, J., Mathieu, E., Beltekian, D. et al. A cross-country
database of COVID-19 testing. Sci Data 7, 345 (2020).
https://doi.org/10.1038/s41597-020-00688-8

-  The number of tests
-  The number of vaccinations

`World Bank Open Data <https://data.worldbank.org/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Citation: World Bank Group (2020), World Bank Open Data,
https://data.worldbank.org/

-  Population pyramid

`Datasets for CovsirPhy <https://github.com/lisphilar/covid19-sir/tree/master/data>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lisphilar (2020), GitHub repository, COVID-19 dataset in Japan.

-  The number of cases in Japan (total/prefectures)
-  Metadata

2. How to request new data loader
---------------------------------

If you want to use a new dataset for your analysis, please kindly inform
us via `GitHub Issues: Request new method of DataLoader
class <https://github.com/lisphilar/covid19-sir/issues/new/?template=request-new-method-of-dataloader-class.md>`__.
Please read `Guideline of
contribution <https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html>`__
in advance.

3. Use a local CSV file which has the number of cases
-----------------------------------------------------

We can replace ``jhu_data`` instance created by ``DataLoader`` class
with your dataset saved in a CSV file.

As an example, we have a CSV file ("oslo.csv") with the following
columns.

-  "date": reported dates
-  "confirmed": the number of confirmed cases
-  "recovered": the number of recovered cases
-  "fatal": the number of fatal cases
-  "province": province names

Please create ``CountryData`` instance at first.

.. code:: python

    # Create CountryData instance
    country_data = cs.CountryData("oslo.csv", country="Norway")
    country_data.set_variables(
        date="date", confirmed="confirmed", recovered="recovered", fatal="fatal", province="province",
    )
    # If you do not have province column, you can specify with province argument
    # country_data = cs.CountryData("oslo.csv", country="Norway", province="Oslo")
    # country_data.set_variables(
    #     date="date", confirmed="confirmed", recovered="recovered", fatal="fatal",
    # )
    # If the dataset does not have province-level records,
    # country_data = cs.CountryData("oslo.csv", country="Norway")
    # country_data.set_variables(
    #     date="date", confirmed="confirmed", recovered="recovered", fatal="fatal",
    # )

Then, convert it to ``JHUData`` instance.

.. code:: python

    # Create JHUData instance using cleaned dataset (pandas.DataFrame)
    jhu_data = cs.JHUData.from_dataframe(country_data.cleaned())
    # Or, we can use and update the output of DataLoader.jhu()
    # jhu_data = data_loader.jhu()
    # jhu_data.replace(country_data)

Additionally, you may need to register population values to
``PopulationData`` instance manually.

.. code:: python

    # Create PopulationData instance with empty dataset
    population_data = cs.PopulationData()
    # Or, we can use the output of DataLoader.population()
    # population_data = data_loader.population()
    # Update the population value
    population_data.update(693494, country="Norway", province="Oslo")

Notes: This is also effective in `Kaggle <https://www.kaggle.com/>`__
Notebook. The datasets are saved in "/kaggle/input/" directory.

Notes: If you have Kaggle API, you can download Kaggle datasets by
updating and executing
`input.py <https://github.com/lisphilar/covid19-sir/blob/master/input.py>`__
script. CSV files will be saved in "/kaggle/input/" directory.

Kaggle API: Move to account page of Kaggle and download "kaggle.json" by
selecting "API > Create New API Token" button. Copy the json file to the
top directory of the local repository or "~/.kaggle". Please refer to
`How to Use Kaggle: Public API <https://www.kaggle.com/docs/api>`__ and
`stackoverflow: documentation for Kaggle API *within*
python? <https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python#:~:text=Here%20are%20the%20steps%20involved%20in%20using%20the%20Kaggle%20API%20from%20Python.&text=Go%20to%20your%20Kaggle%20account,json%20will%20be%20downloaded>`__

Notes: CovsirPhy project started in Kaggle platform with the following
datasets.

-  The number of cases (JHU) and linelist: `Novel Corona Virus 2019
   Dataset by
   SRK <https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset>`__
-  Population in each country: `covid19 global forecasting: locations
   population by Dmitry A.
   Grechka <https://www.kaggle.com/dgrechka/covid19-global-forecasting-locations-population>`__
-  The number of cases in Japan: `COVID-19 dataset in Japan by
   Lisphilar <https://www.kaggle.com/lisphilar/covid19-dataset-in-japan>`__
