| |PyPI version| |Downloads| |PyPI - Python Version| |Build Status|
| |GitHub license| |Maintainability| |Test Coverage|

CovsirPhy introduction
======================

`Documentation <https://lisphilar.github.io/covid19-sir/index.html>`__
\|
`Installation <https://lisphilar.github.io/covid19-sir/INSTALLATION.html>`__
\| `Quickest
usage <https://lisphilar.github.io/covid19-sir/usage_quickest.html>`__
\| `API
reference <https://lisphilar.github.io/covid19-sir/covsirphy.html>`__ \|
`Qiita (Japanese) <https://qiita.com/tags/covsirphy>`__

CovsirPhy is a Python package for COVID-19 (Coronavirus disease 2019)
data analysis with phase-dependent SIR-derived ODE models. We can
download datasets and analyse it easily. This will be a helpful tool for
data-informed decision making. Please refer to "Method" part of `Kaggle
Notebook: COVID-19 data with SIR
model <https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model>`__
to understand the methods.

Functionalities
---------------

-  Data preparation and data visualization
-  `Phase setting with S-R Trend
   analysis <https://lisphilar.github.io/covid19-sir/usage_phases.html>`__
-  Numerical simulation of ODE models

   -  Stable: SIR, SIR-D and SIR-F model
   -  Development: SIR-FV and SEWIR-F model

-  Phase-dependent parameter estimation of ODE models
-  Scenario analysis: Simulate the number of cases with user-defined
   parameter values
-  (In development): Find the relationship of government response and
   parameter values

Inspiration
-----------

-  Monitor the spread of COVID-19
-  Keep track parameter values/reproduction number in each
   country/province
-  Find the relationship of reproductive number and measures taken by
   each country

If you have ideas or need new functionalities, please join this project.
Any suggestions with `Github
Issues <https://github.com/lisphilar/covid19-sir/issues/new/choose>`__
are always welcomed. Please read `Guideline of
contribution <https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html>`__
in advance.

Installation and dataset preparation
------------------------------------

We have the following options to start analysis with CovsirPhy. Datasets
are not included in this package, but we can prepare them with
``DataLoader`` class.

+------------------------+---------------+-------------------------------------+
|                        | Installation  | Dataset preparation                 |
+========================+===============+=====================================+
| Standard users         | pip/pipenv    | Automated with ``DataLoader`` class |
+------------------------+---------------+-------------------------------------+
| Developers             | git-cloning   | Automated with ``DataLoader`` class |
+------------------------+---------------+-------------------------------------+
| Kagglers (local        | git-cloning   | Kaggle API and Python script and    |
| environment)           |               | ``DataLoader``                      |
+------------------------+---------------+-------------------------------------+
| Kagglers (Kaggle       | pip           | Kaggle Datasets and ``DataLoader``  |
| platform)              |               |                                     |
+------------------------+---------------+-------------------------------------+

\ `Installation and dataset
preparation <https://lisphilar.github.io/covid19-sir/INSTALLATION.html>`__
explains how to install and prepare datasets for all users.

Stable versions of Covsirphy are available at `PyPI (The Python Package
Index): covsirphy <https://pypi.org/project/covsirphy/>`__ and support
Python 3.6 or newer versions.

::

    pip install covsirphy --upgrade

Development versions are in `GitHub repository:
CovsirPhy <https://github.com/lisphilar/covid19-sir>`__.

::

    pip install "git+https://github.com/lisphilar/covid19-sir.git#egg=covsirphy"

| Main datasets will be retrieved via `COVID-19 Data
  Hub <https://covid19datahub.io/>`__ and the citation is
| Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub", Journal of Open
  Source Software 5(51):2376, doi: 10.21105/joss.02376.

Usage
-----

Quickest tour of CovsirPhy is here. The following codes analyze the
records in Japan, but we can change the country name when creating
``Scenario`` class instance for your own analysis.

.. code:: python

    import covsirphy as cs
    # Download datasets
    data_loader = cs.DataLoader("input")
    jhu_data = data_loader.jhu()
    population_data = data_loader.population()
    # Check records
    snl = cs.Scenario(jhu_data, population_data, country="Japan")
    snl.records()
    # S-R trend analysis
    snl.trend().summary()
    # Parameter estimation of SIR-F model
    snl.estimate(cs.SIRF)
    # History of reproduction number
    _ = snl.history(target="Rt")
    # History of parameters
    _ = snl.history_rate()
    _ = snl.history(target="rho")
    # Simulation for 30 days
    snl.add(days=30)
    _ = snl.simulate()

Further information:

-  `Quickest
   version <https://lisphilar.github.io/covid19-sir/usage_quickest.html>`__
-  `Quick
   version <https://lisphilar.github.io/covid19-sir/usage_quick.html>`__
-  `Details:
   phases <https://lisphilar.github.io/covid19-sir/usage_phases.html>`__
-  `Details: theoretical
   datasets <https://lisphilar.github.io/covid19-sir/usage_theoretical.html>`__
-  `Details: policy
   measures <https://lisphilar.github.io/covid19-sir/usage_policy.html>`__
-  Example codes in `"example" directory of this
   repository <https://github.com/lisphilar/covid19-sir/tree/master/example>`__
-  `Kaggle: COVID-19 data with SIR
   model <https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model>`__

Support
-------

Please support this project as a developer (or a backer). |Become a
backer|

License: Apache License 2.0
---------------------------

Please refer to
`LICENSE <https://github.com/lisphilar/covid19-sir/blob/master/LICENSE>`__
file.

Citation
--------

We have no original papers the author and contributors wrote, but please
cite this package as follows.

CovsirPhy Development Team (2020), CovsirPhy, Python package for
COVID-19 analysis with SIR-derived ODE models,
https://github.com/lisphilar/covid19-sir

If you want to use SIR-F/SIR-FV/SEWIR-F model, S-R trend analysis,
phase-dependent approach to SIR-derived models, and other scientific
method performed with CovsirPhy, please cite the next Kaggle notebook.

Lisphilar (2020), Kaggle notebook, COVID-19 data with SIR model,
https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

Related work
------------

| Reproduction number evolution in each country:
| Ilyass Tabiai and Houda Kaddioui (2020), GitHub pages, COVID19 R0
  tracker, https://ilylabs.github.io/projects/COVID-trackers/

.. |PyPI version| image:: https://badge.fury.io/py/covsirphy.svg
   :target: https://badge.fury.io/py/covsirphy
.. |Downloads| image:: https://pepy.tech/badge/covsirphy
   :target: https://pepy.tech/project/covsirphy
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/covsirphy
   :target: https://badge.fury.io/py/covsirphy
.. |Build Status| image:: https://semaphoreci.com/api/v1/lisphilar/covid19-sir/branches/master/shields_badge.svg
   :target: https://semaphoreci.com/lisphilar/covid19-sir
.. |GitHub license| image:: https://img.shields.io/github/license/lisphilar/covid19-sir
   :target: https://github.com/lisphilar/covid19-sir/blob/master/LICENSE
.. |Maintainability| image:: https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/maintainability
   :target: https://codeclimate.com/github/lisphilar/covid19-sir/maintainability
.. |Test Coverage| image:: https://api.codeclimate.com/v1/badges/eb97eaf9804f436062b9/test_coverage
   :target: https://codeclimate.com/github/lisphilar/covid19-sir/test_coverage
.. |Become a backer| image:: https://opencollective.com/covsirphy/tiers/backer.svg?avatarHeight=36&width=600
   :target: https://opencollective.com/covsirphy
