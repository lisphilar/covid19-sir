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
CovsirPhy enables us to make data-informed decisions.

Functionalities
---------------

-  `Data preparation and data
   visualization <https://lisphilar.github.io/covid19-sir/usage_dataset.html>`__
-  `Phase setting with S-R Trend
   analysis <https://lisphilar.github.io/covid19-sir/usage_phases.html>`__
-  `Numerical simulation of ODE
   models <https://lisphilar.github.io/covid19-sir/usage_theoretical.html>`__:
   SIR, SIR-D and SIR-F model
-  `Phase-dependent parameter estimation of ODE
   models <https://lisphilar.github.io/covid19-sir/usage_quickest.html>`__
-  `Scenario
   analysis <https://lisphilar.github.io/covid19-sir/usage_quick.html>`__:
   Simulate the number of cases with user-defined parameter values
-  `Find the relationship of government response and parameter
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
are always welcomed. Questions are also great. Please read `Guideline of
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
    # Select country name and register the data
    snl = cs.Scenario(country="Japan")
    snl.register(jhu_data)
    # Check records
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
-  `Kaggle: COVID-19 data with SIR
   model <https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model>`__

Release notes
-------------

Release notes are
`here <https://github.com/lisphilar/covid19-sir/releases>`__. Titles &
links of issues are listed with acknowledgement.

We can see the release plan for the next stable version in `milestone
page of the GitHub
repository <https://github.com/lisphilar/covid19-sir/milestones>`__. If
you find a highly urgent matter, please let us know via `issue
page <https://github.com/lisphilar/covid19-sir/issues>`__.

Support
-------

Please support this project as a developer (or a backer). |Become a
backer|

Developers
----------

CovsirPhy library is developed by a community of volunteers. Please see
the full list
`here <https://github.com/lisphilar/covid19-sir/graphs/contributors>`__.

This project started in Kaggle platform. Lisphilar published `Kaggle
Notebook: COVID-19 data with SIR
model <https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model>`__
on 12Feb2020 and developed it, discussing with Kaggle community. On
07May2020, "covid19-sir" repository was created. On 10May2020,
``covsirphy`` version 1.0.0 was published in GitHub. First release in
PyPI (version 2.3.0) was on 28Jun2020.

License: Apache License 2.0
---------------------------

Please refer to
`LICENSE <https://github.com/lisphilar/covid19-sir/blob/master/LICENSE>`__
file.

Citation
--------

We have no original papers the author and contributors wrote, but please
cite this library as follows with version number
(``import covsirphy as cs; cs.__version__``).

CovsirPhy Development Team (2020-2021), CovsirPhy version [version
number]: Python library for COVID-19 analysis with phase-dependent
SIR-derived ODE models, https://github.com/lisphilar/covid19-sir

If you want to use SIR-F model, S-R trend analysis, phase-dependent
approach to SIR-derived models, and other scientific method performed
with CovsirPhy, please cite the next Kaggle notebook.

Hirokazu Takaya (2020-2021), Kaggle Notebook, COVID-19 data with SIR
model, https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

We can check the citation with the following script.

.. code:: python

    import covsirphy as cs
    cs.__citation__

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
