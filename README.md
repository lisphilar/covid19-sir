
<img src="https://raw.githubusercontent.com/lisphilar/covid19-sir/master/docs/logo/covsirphy_headline.png" width="390" alt="CovsirPhy: COVID-19 analysis with phase-dependent SIRs">

[![PyPI version](https://badge.fury.io/py/covsirphy.svg)](https://badge.fury.io/py/covsirphy)
[![Downloads](https://pepy.tech/badge/covsirphy)](https://pepy.tech/project/covsirphy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/covsirphy)](https://badge.fury.io/py/covsirphy)
[![GitHub license](https://img.shields.io/github/license/lisphilar/covid19-sir)](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE)
[![Quality Check](https://github.com/lisphilar/covid19-sir/actions/workflows/test.yml/badge.svg)](https://github.com/lisphilar/covid19-sir/actions/workflows/test.yml)
[![Test Coverage](https://codecov.io/gh/lisphilar/covid19-sir/branch/master/graph/badge.svg?token=9Z8Z1UHY3I)](https://codecov.io/gh/lisphilar/covid19-sir)

# CovsirPhy introduction

[<strong>Documentation</strong>](https://lisphilar.github.io/covid19-sir/index.html)
| [<strong>Installation</strong>](https://lisphilar.github.io/covid19-sir/markdown/INSTALLATION.html)
| [<strong>Tutorial</strong>](<https://lisphilar.github.io/covid19-sir/01_data_preparation.html>)
| [<strong>API reference</strong>](https://lisphilar.github.io/covid19-sir/covsirphy.html)
| [<strong>GitHub</strong>](https://github.com/lisphilar/covid19-sir)
| [<strong>Qiita (Japanese)</strong>](https://qiita.com/tags/covsirphy)

<strong>CovsirPhy is a Python library for infectious disease (COVID-19: Coronavirus disease 2019, Monkeypox 2022) data analysis with phase-dependent SIR-derived ODE models. We can download datasets and analyze them easily. Scenario analysis with CovsirPhy enables us to make data-informed decisions. </strong>

## Inspiration

* Monitor the spread of COVID-19/Monkeypox with SIR-derived ODE models
* Predict the number of cases in each country/province
* Find the relationship of reproductive number and measures taken by each country

<strong>If you have ideas or need new functionalities, please join this project.
Any suggestions with [Github Issues](https://github.com/lisphilar/covid19-sir/issues/new/choose) and [Twitter: @lisphilar](https://twitter.com/lisphilar) are always welcomed. Questions are also great. Please refer to [Guideline of contribution](https://lisphilar.github.io/covid19-sir/CONTRIBUTING.html).</strong>

## Installation

The latest stable version of CovsirPhy is available at [PyPI (The Python Package Index): covsirphy](https://pypi.org/project/covsirphy/) and supports Python 3.8 or newer versions. Details are explained in [Documentation: Installation](https://lisphilar.github.io/covid19-sir/INSTALLATION.html).

```Bash
pip install --upgrade covsirphy
```

> **Warning**  
> We cannot use `covsirphy` on Google Colab, which uses Python 3.7. [Binder](https://mybinder.org/) is recommended.

## Demo

Quickest tour of CovsirPhy is here. The following codes analyze the records in Japan.

```Python
import covsirphy as cs
# Data preparation,time-series segmentation, parameter estimation with SIR-F model
snr = cs.ODEScenario.auto_build(geo="Japan", model=cs.SIRFModel)
# Check actual records
snr.simulate(name=None);
# Show the result of time-series segmentation
snr.to_dynamics(name="Baseline").detect();
# Perform simulation with estimated ODE parameter values
snr.simulate(name="Baseline");
# Predict ODE parameter values (30 days from the last date of actual records)
snr.build_with_template(name="Predicted", template="Baseline");
snr.predict(days=30, name="Predicted");
# Perform simulation with estimated and predicted ODE parameter values
snr.simulate(name="Predicted");
# Add a future phase to the baseline (ODE parameters will not be changed)
snr.append();
# Show created phases and ODE parameter values
snr.summary()
# Compare reproduction number of scenarios (predicted/baseline)
snr.compare_param("Rt");
# Compare simulated number of cases
snr.compare_cases("Confirmed");
# Describe representative values
snr.describe()
```

Output of `snr.simulate(name="Predicted");`

<img src="https://raw.githubusercontent.com/lisphilar/covid19-sir/master/example/output/demo_jpn/04_predicted.png" width="600">

## Tutorial

Tutorials of functionalities are included in the [CovsirPhy documentation](https://lisphilar.github.io/covid19-sir/index.html).

* [Data preparation](https://lisphilar.github.io/covid19-sir/01_data_preparation.html)
* [Data Engineering](https://lisphilar.github.io/covid19-sir/02_data_engineering.html)
* [SIR-derived ODE models](https://lisphilar.github.io/covid19-sir/03_ode.html)
* [Phase-dependent SIR models](https://lisphilar.github.io/covid19-sir/04_phase_dependent.html)
* [Scenario analysis](https://lisphilar.github.io/covid19-sir/05_scenario_analysis.html)
* [ODE parameter prediction](https://lisphilar.github.io/covid19-sir/06_prediction.html)

## Release notes

Release notes are [here](https://github.com/lisphilar/covid19-sir/releases). Titles & links of issues are listed with acknowledgement.

We can see the release plan for the next stable version in [milestone page of the GitHub repository](https://github.com/lisphilar/covid19-sir/milestones). If you find a highly urgent matter, please let us know via [issue page](https://github.com/lisphilar/covid19-sir/issues).

## Developers

CovsirPhy library is developed by a community of volunteers. Please see the full list [here](https://github.com/lisphilar/covid19-sir/graphs/contributors).

This project started in Kaggle platform. Hirokazu Takaya ([@lisphilar](<https://www.kaggle.com/lisphilar>)) published [Kaggle Notebook: COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model) on 12Feb2020 and developed it, discussing with Kaggle community. On 07May2020, "covid19-sir" repository was created. On 10May2020, `covsirphy` version 1.0.0 was published in GitHub. First release in PyPI (version 2.3.0) was on 28Jun2020.

## Support

Please support this project as a developer (or a backer).
[![Become a backer](https://opencollective.com/covsirphy/tiers/backer.svg?avatarHeight=36&width=600)](https://opencollective.com/covsirphy)

## License: Apache License 2.0

Please refer to [LICENSE](https://github.com/lisphilar/covid19-sir/blob/master/LICENSE) file.

## Citation

Please cite this library as follows with version number (`import covsirphy as cs; cs.__version__`).

**Hirokazu Takaya and CovsirPhy Development Team (2020-2022), CovsirPhy version [version number]: Python library for COVID-19 analysis with phase-dependent SIR-derived ODE models, [https://github.com/lisphilar/covid19-sir](https://github.com/lisphilar/covid19-sir)**

This is the output of `covsirphy.__citation__`.

```Python
import covsirphy as cs
cs.__citation__
```

**We have no original papers the author and contributors wrote, but note that some scientific approaches, including SIR-F model, S-R change point analysis, phase-dependent approach to SIR-derived models, were developed in this project.**
