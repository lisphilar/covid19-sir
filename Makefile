.PHONY: poetry-linux
poetry-linux:
	@# Install poetry (Linux, OSX, WSL)
	@# system Python should be installed in advance
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
	@poetry --version
	@poetry config virtualenvs.in-project true
	@poetry config repositories.testpypi https://test.pypi.org/legacy/
	@poetry config --list

.PHONY: poetry-windows
poetry-windows:
	@# Install poetry (Windows)
	@# system Python should be installed in advance
	@(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
	@poetry --version
	@poetry config virtualenvs.in-project true
	@poetry config repositories.testpypi https://test.pypi.org/legacy/
	@poetry config --list

.PHONY: install
install:
	@pip install --upgrade pip
	@poetry self update
	@poetry install

.PHONY: update
update:
	@poetry update

.PHONY: test
test:
	@# All tests: make test
	@# Selected tests: make test target=/test_scenario.py::TestScenario
	@poetry run flake8 covsirphy --ignore=E501
	@poetry run pytest tests${target} -v --durations=0 --failed-first --maxfail=1 \
	 --cov=covsirphy --cov-report=term-missing

# https://github.com/sphinx-doc/sphinx/issues/3382
.PHONY: sphinx
sphinx:
	@# sudo apt install pandoc
	@# update docs/index.rst as well as the following codes
	@pandoc --from markdown --to rst README.md -o docs/README.rst
	@pandoc --from markdown --to rst .github/CODE_OF_CONDUCT.md -o docs/CODE_OF_CONDUCT.rst
	@pandoc --from markdown --to rst .github/CONTRIBUTING.md -o docs/CONTRIBUTING.rst
	@pandoc --from markdown --to rst SECURITY.md -o docs/SECURITY.rst
	@pandoc --from markdown --to rst docs/markdown/INSTALLATION.md -o docs/INSTALLATION.rst
	@pandoc --from markdown --to rst docs/markdown/TERM.md -o docs/TERM.rst
	@# When new module was added, update docs/covsirphy.rst and docs/(module name).rst
	@sphinx-apidoc -o docs covsirphy -fMT
	@cd docs; poetry run make html; cp -a _build/html/. ../docs
	@rm -rf docs/_modules
	@rm -rf docs/_sources

# https://github.com/sphinx-doc/sphinx/issues/3382
.PHONY: docs
docs:
	@rm -rf docs/_images
	@rm -f docs/*ipynb
	@# docs/index.rst must be updated to include the notebooks
	@poetry run runipy example/usage_quick.ipynb docs/usage_quick.ipynb
	@poetry run runipy example/usage_dataset.ipynb docs/usage_dataset.ipynb
	@poetry run runipy example/usage_quickest.ipynb docs/usage_quickest.ipynb
	@poetry run runipy example/usage_phases.ipynb docs/usage_phases.ipynb
	@poetry run runipy example/usage_theoretical.ipynb docs/usage_theoretical.ipynb
	@poetry run runipy example/usage_policy.ipynb docs/usage_policy.ipynb
	@make sphinx

.PHONY: pypi
pypi:
	@rm -rf covsirphy.egg-info/*
	@rm -rf dist/*
	@poetry publish --build

.PHONY: test-pypi
test-pypi:
	@rm -rf covsirphy.egg-info/*
	@rm -rf dist/*
	@poetry publish -r testpypi --build

.PHONY: clean
clean:
	@rm -rf input
	@rm -rf kaggle
	@rm -rf prof
	@rm -rf .pytest_cache
	@find -name __pycache__ | xargs --no-run-if-empty rm -r
	@rm -rf example/output
	@rm -rf dist covsirphy.egg-info
	@rm -f README.rst
	@rm -f .coverage*
	@pip install --upgrade pip
	@poetry self update
	@poetry update
