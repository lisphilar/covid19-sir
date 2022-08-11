# Installation

`covsirphy` library supports Python 3.7.12 and newer versions.

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

## Start analysis

What to do next? Please refer to tutorials of this documentation. The first tutorial is [Data preparation](https://lisphilar.github.io/covid19-sir/01_data_preparation.html). If you have any questions and any advice, feel free to contact developers and users via [GitHub Issues](https://github.com/lisphilar/covid19-sir/issues).
