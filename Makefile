.PHONY: test docs example clean

test:
	@pipenv run pytest -v --durations=0 --profile-svg


# https://github.com/sphinx-doc/sphinx/issues/3382
docs:
	@sphinx-apidoc -o docs covsirphy
	@cd docs; pipenv run make html; cp -a _build/html/. ../docs


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


clean:
	@rm -rf input
	@rm -rf prof
	@rm -rf .pytest_cache
	@rm -rf covsirphy/__pycache__
	@pipenv clean || true
