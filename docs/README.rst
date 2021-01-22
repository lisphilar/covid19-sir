| |PyPI version| |Downloads| |PyPI - Python Version| |Build Status|
| |GitHub license| |Maintainability| |Test Coverage| |Open Source
  Helpers|

CovsirPhy introduction
======================

`Documentation <https://lisphilar.github.io/covid19-sir/index.html>`__
\|
`Installation <https://lisphilar.github.io/covid19-sir/INSTALLATION.html>`__
\| `Quickest
usage <https://lisphilar.github.io/covid19-sir/usage_quickest.html>`__
\| `API
reference <https://lisphilar.github.io/covid19-sir/covsirphy.html>`__ \|
`GitHub <https://github.com/lisphilar/covid19-sir>`__ \| `Qiita
(Japanese) <https://qiita.com/tags/covsirphy>`__

CovsirPhy is a Python library for COVID-19 (Coronavirus disease 2019)
data analysis with phase-dependent SIR-derived ODE models. We can
download datasets and analyse them easily. Scenario analysis with
CovsirPhy enables us to make data-informed decisions. Please refer to
"Method" part of `Kaggle Notebook: COVID-19 data with SIR
model <https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model>`__
to understand the methods.

Functionalities
---------------

-  `Data preparation and data
   visualization <https://lisphilar.github.io/covid19-sir/usage_dataset.html>`__
-  `Phase setting with S-R Trend
   analysis <https://lisphilar.github.io/covid19-sir/usage_phases.html>`__
-  `Numerical simulation of ODE
   models <https://lisphilar.github.io/covid19-sir/usage_theoretical.html>`__
-  Stable: SIR, SIR-D and SIR-F model
-  Development: SIR-FV and SEWIR-F model
-  `Phase-dependent parameter estimation of ODE
   models <https://lisphilar.github.io/covid19-sir/usage_quickest.html>`__
-  `Scenario
   analysis <https://lisphilar.github.io/covid19-sir/usage_quick.html>`__:
   Simulate the number of cases with user-defined parameter values
-  `(In development): Find the relationship of government response and
   parameter
   values <https://lisphilar.github.io/covid19-sir/usage_policy.html>`__

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

Installation
------------

The latest stable version of CovsirPhy is available at `PyPI (The Python
Package Index): covsirphy <https://pypi.org/project/covsirphy/>`__ and
supports Python 3.7 or newer versions. Details are explained in
`Documentation:
Installation <https://lisphilar.github.io/covid19-sir/INSTALLATION.html>`__.

.. code:: bash

    pip install --upgrade covsirphy

Usage
-----

Quickest tour of CovsirPhy is here. The following codes analyze the
records in Japan, but we can change the country name when creating
``Scenario`` class instance for your own analysis.

.. code:: python

    import covsirphy as cs
    # Download and update datasets
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

-  `CovsirPhy
   documentation <https://lisphilar.github.io/covid19-sir/index.html>`__
-  Example scripts in `"example" directory of this
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
.. |Open Source Helpers| image:: https://www.codetriage.com/lisphilar/covid19-sir/badges/users.svg
   :target: https://www.codetriage.com/lisphilar/covid19-sir
.. |Become a backer| image:: https://opencollective.com/covsirphy/tiers/backer.svg?avatarHeight=36&width=600
   :target: https://opencollective.com/covsirphy
