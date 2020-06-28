.PHONY: test docs example pypi test-pypi clean

install:
	@pip install wheel; pip install --upgrade pip
	@pip install pipenv
	@export PIPENV_VENV_IN_PROJECT=true
	@export PIPENV_TIMEOUT=7200
	@pipenv install --dev


test:
	@pipenv run pytest -v --durations=0 --profile-svg


# https://github.com/sphinx-doc/sphinx/issues/3382
docs:
	@# sudo apt install pandoc
	@pandoc --from markdown --to rst README.md -o docs/README.rst
	@sphinx-apidoc -o docs covsirphy
	@cd docs; pipenv run make html; cp -a _build/html/. ../docs
	@rm -rf docs/_modules
	@rm -rf docs/_sources


example:
	@# Data cleaning
	@echo "<Data loading>"
	@pipenv run python -m example.dataset_load

	@# ODE simulation and hyperparameter estimation
	@echo "<ODE simulation and hyperparameter estimation>"
	@echo "SIR model"
	@pipenv run python -m example.sir_model
	@echo "SIR-D model"
	@pipenv run python -m example.sird_model
	@echo "SIR-F model"
	@pipenv run python -m example.sirf_model
	@echo "SIR-FV model"
	@pipenv run python -m example.sirfv_model
	@echo "SEWIR-F model"
	@pipenv run python -m example.sewirf_model

	@# Long ODE simulation with SIR-F model
	@echo "<Long ODE simulation with SIR-F model>"
	@pipenv run python -m example.long_simulation

	@# Reproductive hyperparameter estimation
	@echo "<Reproductive hyperparameter estimation>"
	@pipenv run python -m example.reproductive_optimization

	@# Scenario analysis
	@echo "<Scenario analysis>"
	@pipenv run python -m example.scenario_analysis

pypi:
	@# sudo apt install pandoc
	@pandoc --from markdown --to rst README.md -o README.rst
	@rm -rf covsirphy.egg-info/*
	@rm -rf dist/*
	@pipenv run python setup.py sdist bdist_wheel
	@pipenv run twine upload --repository pypi dist/*


test-pypi:
	@# sudo apt install pandoc
	@pandoc --from markdown --to rst README.md -o README.rst
	@rm -rf covsirphy.egg-info/*
	@rm -rf dist/*
	@pipenv run python setup.py sdist bdist_wheel
	@pipenv run twine upload --repository testpypi dist/*


clean:
	@rm -rf input
	@rm -rf prof
	@rm -rf .pytest_cache
	@rm -rf covsirphy/__pycache__
	@rm -rf example/output
	@rm -rf dist CovsirPhy.egg-info
	@rm -f README.rst
	@pipenv clean || true
