# Guideline of contribution
Thank you always for using CovsirPhy and continuous supports!  
We hope this project contribute to COVID-19 outbreak research and we will overcome the outbreak in the near future.

When you contribute to [covid-sir repository](https://github.com/lisphilar/covid19-sir) including CovsirPhy package, please kindly follow [Code of conduct](https://lisphilar.github.io/covid19-sir/CODE_OF_CONDUCT.html) and the steps explained in this document. If you have any questions or any request to change about this guideline, please inform the author via issue page.

Terms:

- 'Maintainers' (author/collaborators) maintain this repository. i.e. Merge pull requests, publish [sphinx document](https://lisphilar.github.io/covid19-sir/) and update the version number.
- 'Developers' create issues/pull requests, lead discussions, and update documentation.
- 'Users' uses this package for analysing the datasets, publish the outputs, create issues, join discussions.

## Clone CovsirPhy repository and install dependencies
Developers will clone CovsirPhy repository with `git clone` command and install dependencies with Poetry.

### Clone/Fork the repository
Please start by cloning the GitHub repository with git command.

```
git clone https://github.com/lisphilar/covid19-sir.git
cd covid19-sir
```

Or, fork the repository and run the following commands with your account name `<your account>`. "remotes/upstream/master" will be shown by `git branch -a`.

```
git clone https://github.com/<your account>/covid19-sir.git
git remote add upstream https://github.com/lisphilar/covid19-sir.git
git branch -a
```

Before editing the codes, please fetch and merge the upstream.

```
git fetch upstream
git checkout master
git merge upstream/master
```

### Setup Python
Setup base Python with https://www.python.org/downloads/ (Windows) or some commands (anyenv and so on, Linux/WSL/macOS).

### Install Poetry (Windows)
Then, please install Poetry, a package management tool, with command lien tools, including PowerShell.

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
poetry --version
poetry config virtualenvs.in-project true
poetry config --list
```

If you have `make` commands, run `make poetry-windows`. This is defined in "Makefile" of the top directory.

### Install Poetry (Linux, WSL, macOS)
Please use `make` command.

```
make poetry-linux
```

### Install dependencies
To install dependencies with Poetry, run `make install` or the following commands before editing codes.

```
pip install --upgrade 
poetry self update
poetry install
```

- New dependencies can be installed with `poetry add <name>` or `poetry add <name> --dev`.
- Un-necessary dependencies will be removed with `poetry remove <name>` or `poetry remove <name> --dev`.
- Start shell with `poetry shell; python`.
- Run python scripts with `poetry run` command, like `poetry run examples/scenario_analysis.py`.

## Issues
Kindly [create an issue](https://github.com/lisphilar/covid19-sir/issues) when you want to

- ask a question
- request fixing a bug,
- request updating documentation,
- request a new feature,
- request a new data loader or new dataset,
- communicate with the maintainers and the user community.

Please ensure that the issue

- has one question/request,
- uses appropriate issue template if available,
- has appropriate title, and
- explains your execution environment and codes.

When requesting a new data loader, please confirm that the dataset follows ALCOA (Attributable, Legible, Contemporaneous, Original, Accurate) and we can retrieve the data with permitted APIs. If the data loaders included in this repository use un-permitted APIs, please inform maintainers. We will fix it as soon as possible.

## Pull request
If you have any ideas, please [create a pull request](https://github.com/lisphilar/covid19-sir/pulls) after explaining your idea in an issue.
Please ensure that

- you are using the latest version of CovsirPhy,
- the codes are readable,
- (if necessary) test codes are added,
- tests were successfully completed as discussed later, and
- the revision is well documented in docstring (Google-style)/README and so on.

Please use appropriate template if available, and specify the related issues and the change.

We use GitHub flow here. Default branch can be deployed always as the latest development version. Name of branch for pull requests will be linked to issue numbers, like "issue100".

To update tutorials in documentation, please update ".ipynb" (Jupyter Notebook) files in "example" directory.

Update of version number and sphinx documents is not necessary for developers. After reviewing by assigned reviewers, maintainers will update them by editing "covsirphy/\_\_version\_\_.py" and "pyproject.toml" when merge the pull request.

## Run tests
Before creating a pull request, please run tests with `make pytest` or the following commands.

All tests:

```Python
poetry run flake8 covsirphy --ignore=E501
poetry run pytest tests -v --durations=0 --failed-first --maxfail=1 --cov=covsirphy --cov-report=term-missing
```

Selected tests:
(e.g. when you updated codes related to tests/test_scenario.py)

Run `make test target=/test_scenario.py` or commands as follows.

```Python
poetry run flake8 covsirphy --ignore=E501
poetry run pytest tests/test_scenario.py -v --durations=0 --failed-first --maxfail=1 \
    --cov=covsirphy --cov-report=term-missing
```

When you create a pull request to upstream repository, CI tools will test the codes with Python 3.7 and 3.8. When development version number is updated (i.e. a pull request merged), CI tools will test the codes with the all supported Python versions.

## Versioning
CovsirPhy follows [Semantic Versioning 2.0.0](https://semver.org/):

- Milestones of minor update (from X.0.Z to X.1.Z) are documented in [milestones of issues](https://github.com/lisphilar/covid19-sir/milestones).
- Development version number will be updated, e.g. "version 1.0.0-alpha" to "version 1.0.0-beta" (for closing an issue), "version 1.0.0-better-fu1" (for follow-up).
- When the revisions do not change the codes of CovsirPhy, version number will not be updated.

Maintainers will

- update [sphinx document](https://lisphilar.github.io/covid19-sir/) with CI tools and `make docs`,
- update "pyproject.toml" and "poetry.lock" with `make update`,
- update "covsirphy/\_\_init\_\_.py" to update development or stable version number,
- update "pyproject.toml" to update stable version number,
- upload to [PyPI: The Python Package Index](https://pypi.org/), and
- create [a release note](https://github.com/lisphilar/covid19-sir/releases) for major/minor/batch update.
