Installation and dataset preparation
====================================

We have the following options to start analysis with CovsirPhy. Datasets
are not included in this package, but we can prepare them with
``DataLoader`` class.

+--------------------------------+----------------+---------------------------------------+
|                                | Installation   | Dataset preparation                   |
+================================+================+=======================================+
| Standard users                 | pip/pipenv     | Automated with ``DataLoader`` class   |
+--------------------------------+----------------+---------------------------------------+
| Developers                     | git-cloning    | Automated with ``DataLoader`` class   |
+--------------------------------+----------------+---------------------------------------+
| Kagglers (local environment)   | git-cloning    | Kaggle API and Python script          |
+--------------------------------+----------------+---------------------------------------+
| Kagglers (Kaggle platform)     | pip            | Kaggle Datasets                       |
+--------------------------------+----------------+---------------------------------------+

We will use the following datasets (CovsirPhy >= 2.4.0). Standard users
and developers will retrieve main datasets from `COVID-19 Data
Hub <https://covid19datahub.io/>`__ using ``covid19dh`` Python package.
We can get the citation list of primary source via `COVID-19 Data Hub:
Dataset <https://covid19datahub.io/articles/data.html>`__ and
``covsirphy.DataLoader`` class (refer to "Standard users" subsection).

+-------+-------+-------+
|       | Descr | URL   |
|       | iptio |       |
|       | n     |       |
+=======+=======+=======+
| The   | Guido | https |
| numbe | tti,  | ://co |
| r     | E.,   | vid19 |
| of    | Ardia | datah |
| cases | ,     | ub.io |
| (JHU  | D.,   | /     |
| style | (2020 |       |
| )     | ),    |       |
|       | "COVI |       |
|       | D-19  |       |
|       | Data  |       |
|       | Hub", |       |
|       | Worki |       |
|       | ng    |       |
|       | paper |       |
|       | ,     |       |
|       | doi:  |       |
|       | 10.13 |       |
|       | 140/R |       |
|       | G.2.2 |       |
|       | .1164 |       |
|       | 9.817 |       |
|       | 63.   |       |
+-------+-------+-------+
| The   | Lisph | https |
| numbe | ilar  | ://gi |
| r     | (2020 | thub. |
| of    | ),    | com/l |
| cases | COVID | isphi |
| in    | -19   | lar/c |
| Japan | datas | ovid1 |
|       | et    | 9-sir |
|       | in    | /tree |
|       | Japan | /mast |
|       | .     | er/da |
|       |       | ta    |
+-------+-------+-------+
| Popul | Guido | https |
| ation | tti,  | ://co |
| in    | E.,   | vid19 |
| each  | Ardia | datah |
| count | ,     | ub.io |
| ry    | D.,   | /     |
|       | (2020 |       |
|       | ),    |       |
|       | "COVI |       |
|       | D-19  |       |
|       | Data  |       |
|       | Hub", |       |
|       | Worki |       |
|       | ng    |       |
|       | paper |       |
|       | ,     |       |
|       | doi:  |       |
|       | 10.13 |       |
|       | 140/R |       |
|       | G.2.2 |       |
|       | .1164 |       |
|       | 9.817 |       |
|       | 63.   |       |
+-------+-------+-------+
| Gover | Guido | https |
| nment | tti,  | ://co |
| Respo | E.,   | vid19 |
| nse   | Ardia | datah |
| Track | ,     | ub.io |
| er    | D.,   | /     |
| (OxCG | (2020 |       |
| RT)   | ),    |       |
|       | "COVI |       |
|       | D-19  |       |
|       | Data  |       |
|       | Hub", |       |
|       | Worki |       |
|       | ng    |       |
|       | paper |       |
|       | ,     |       |
|       | doi:  |       |
|       | 10.13 |       |
|       | 140/R |       |
|       | G.2.2 |       |
|       | .1164 |       |
|       | 9.817 |       |
|       | 63.   |       |
+-------+-------+-------+

If you want to use a new dataset for your analysis, please kindly inform
us via `GitHub
Issues <https://github.com/lisphilar/covid19-sir/issues/new/choose>`__
with "Request new method of DataLoader class" template. Please read
`Guideline of contribution <../../../.github/CONTRIBUTING.html>`__ in
advance.

1. Standard users
~~~~~~~~~~~~~~~~~

Covsirphy is available at `PyPI (The Python Package Index):
covsirphy <https://pypi.org/project/covsirphy/>`__ and supports Python
3.7 or newer versions.

::

    pip install covsirphy

Then, download the datasets with the following codes, when you want to
save the data in ``input`` directory.

.. code:: python

    import covsirphy as cs
    data_loader = cs.DataLoader("input")
    jhu_data = data_loader.jhu()
    japan_data = data_loader.japan()
    population_data = data_loader.population()
    oxcgrt_data = data_loader.oxcgrt()

If ``input`` directory has the datasets, ``DataLoader`` will load the
local files. If the datasets were updated in remote servers,
``DataLoader`` will update the local files automatically.

We can get descriptions of the datasets and raw/cleaned datasets easily.
As an example, JHU dataset will be used here.

.. code:: python

    # Description (string)
    jhu_data.citation
    # Raw data (pandas.DataFrame)
    jhu_data.raw
    # Cleaned data (pandas.DataFrame)
    jhu_data.cleaned()

We can get COVID-19 Data Hub citation list of primary sources as
follows.

.. code:: python

    data_loader.covid19dh_citation

2. Developers
~~~~~~~~~~~~~

Developers will clone this repository with ``git clone`` command and
install dependencies with pipenv.

::

    git clone https://github.com/lisphilar/covid19-sir.git
    cd covid19-sir
    pip install wheel; pip install --upgrade pip; pip install pipenv
    export PIPENV_VENV_IN_PROJECT=true
    export PIPENV_TIMEOUT=7200
    pipenv install --dev

Developers can perform tests with
``pipenv run pytest -v --durations=0 --failed-first --profile-svg`` and
call graph will be saved as SVG file (prof/combined.svg).

-  Windows users need to install `Graphviz for
   Windows <https://graphviz.org/_pages/Download/Download_windows.html>`__
   in advance.
-  Debian/Ubuntu users need to install Graphviz with
   ``sudo apt install graphviz`` in advance.

If you can run ``make`` command,

+--------------------+----------------------------------------------------+
| ``make install``   | Install pipenv and the dependencies of CovsirPhy   |
+--------------------+----------------------------------------------------+
| ``make test``      | Run tests using Pytest                             |
+--------------------+----------------------------------------------------+
| ``make docs``      | Update sphinx document                             |
+--------------------+----------------------------------------------------+
| ``make example``   | Run example codes                                  |
+--------------------+----------------------------------------------------+
| ``make clean``     | Clean-up output files and pipenv environment       |
+--------------------+----------------------------------------------------+

We can prepare the dataset with the same codes as that was explained in
"1. Standard users" subsection.

3. Kagglers (local environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in "2. Developers" subsection, we need to git-clone this
repository and install the dependencies when you want to uses this
package with Kaggle API in your local environment.

Then, please move to account page and download "kaggle.json" by
selecting "API > Create New API Token" button. Copy the json file to the
top directory of the local repository. Please refer to `How to Use
Kaggle: Public API <https://www.kaggle.com/docs/api>`__ and
`stackoverflow: documentation for Kaggle API *within*
python? <https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python#:~:text=Here%20are%20the%20steps%20involved%20in%20using%20the%20Kaggle%20API%20from%20Python.&text=Go%20to%20your%20Kaggle%20account,json%20will%20be%20downloaded>`__

We can download datasets with ``pipenv run ./input.py`` command.
Modification of environment variables is un-necessary. Files will be
saved in ``input`` directory of your local repository.

| Note:
| Except for OxCGRT dataset, the datasets downloaded with ``input.py``
  scripts are different from that explained in the previous subsections.
  URLs are shown in the next table.

+-------+-------+-------+
|       | Descr | URL   |
|       | iptio |       |
|       | n     |       |
+=======+=======+=======+
| The   | Novel | https |
| numbe | Coron | ://ww |
| r     | a     | w.kag |
| of    | Virus | gle.c |
| cases | 2019  | om/su |
| (JHU) | Datas | dalai |
|       | et    | rajku |
|       | by    | mar/n |
|       | SRK   | ovel- |
|       |       | coron |
|       |       | a-vir |
|       |       | us-20 |
|       |       | 19-da |
|       |       | taset |
+-------+-------+-------+
| The   | COVID | https |
| numbe | -19   | ://ww |
| r     | datas | w.kag |
| of    | et    | gle.c |
| cases | in    | om/li |
| in    | Japan | sphil |
| Japan | by    | ar/co |
|       | Lisph | vid19 |
|       | ilar  | -data |
|       |       | set-i |
|       |       | n-jap |
|       |       | an    |
+-------+-------+-------+
| Popul | covid | https |
| ation | 19    | ://ww |
| in    | globa | w.kag |
| each  | l     | gle.c |
| count | forec | om/dg |
| ry    | astin | rechk |
|       | g:    | a/cov |
|       | locat | id19- |
|       | ions  | globa |
|       | popul | l-for |
|       | ation | ecast |
|       | by    | ing-l |
|       | Dmitr | ocati |
|       | y     | ons-p |
|       | A.    | opula |
|       | Grech | tion  |
|       | ka    |       |
+-------+-------+-------+
| Gover | Thoma | https |
| nment | s     | ://gi |
| Respo | Hale, | thub. |
| nse   | Sam   | com/O |
| Track | Webst | xCGRT |
| er    | er,   | /covi |
| (OxCG | Anna  | d-pol |
| RT)   | Pethe | icy-t |
|       | rick, | racke |
|       | Toby  | r     |
|       | Phill |       |
|       | ips,  |       |
|       | and   |       |
|       | Beatr |       |
|       | iz    |       |
|       | Kira. |       |
|       | (2020 |       |
|       | ).    |       |
|       | Oxfor |       |
|       | d     |       |
|       | COVID |       |
|       | -19   |       |
|       | Gover |       |
|       | nment |       |
|       | Respo |       |
|       | nse   |       |
|       | Track |       |
|       | er.   |       |
|       | Blava |       |
|       | tnik  |       |
|       | Schoo |       |
|       | l     |       |
|       | of    |       |
|       | Gover |       |
|       | nment |       |
|       | .     |       |
+-------+-------+-------+

Usage of ``DataLoader`` class is as follows. Please specify
``local_file`` argument in the methods.

.. code:: python

    import covsirphy as cs
    data_loader = cs.DataLoader("input")
    jhu_data = data_loader.jhu(local_file="covid_19_data.csv")
    japan_data = data_loader.japan(local_file="covid_jpn_total.csv")
    population_data = data_loader.population(local_file="locations_population.csv")
    oxcgrt_data = data_loader.oxcgrt(local_file="OxCGRT_latest.csv")

4. Kagglers (Kaggle platform)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you want to use this package in Kaggle notebook, please turn on
Internet option in notebook setting and download the datasets explained
in the previous section.

Then, install this package with pip command.

::

    !pip install covsirphy

Then, please load the datasets with the following codes, specifying the
filenames.

.. code:: python

    import covsirphy as cs
    # The number of cases (JHU)
    jhu_data = cs.JHUData("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
    # (Optional) The number of cases in Japan
    japan_data = cs.CountryData("/kaggle/input/covid19-dataset-in-japan/covid_jpn_total.csv", country="Japan")
    japan_data.set_variables(
        date="Date", confirmed="Positive", fatal="Fatal", recovered="Discharged", province=None
    )
    # Population in each country
    population_data = cs.PopulationData(
        "/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv"
    )

| Note:
| Currently, OxCGRT dataset is not supported.
