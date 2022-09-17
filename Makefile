.PHONY: setup-anyenv
setup-anyenv:
	@ # Set-up anyenv in Bash (Linux, WSL)
	@git clone https://github.com/riywo/anyenv ~/.anyenv
	@echo 'export PATH="$HOME/.anyenv/bin:$PATH"' >> ~/.bashrc; source ~/.bashrc
	@anyenv install --init: echo 'eval "$(anyenv init -)"' >> ~/.bashrc; source ~/.bashrc
	@/bin/mkdir -p $(anyenv root)/plugins; git clone https://github.com/znz/anyenv-update.git $(anyenv root)/plugins/anyenv-update
	@ # set-up pyenv
	@anyenv install pyenv;  exec ${SHELL} -l

.PHONY: setup-python
setup-python:
	@# Install the latest stable version of Python and set default for CovsirPhy project
	@anyenv update --force
	@echo python `pyenv install -l | grep -x '  [0-9]*\.[0-9]*\.[0-9]*' | tail -n 1 | tr -d ' '`
	@pyenv install -f `pyenv install -l | grep -x '  [0-9]*\.[0-9]*\.[0-9]*' | tail -n 1 | tr -d ' '`
	@pyenv local `pyenv install -l | grep -x '  [0-9]*\.[0-9]*\.[0-9]*' | tail -n 1 | tr -d ' '`
	@anyenv versions

.PHONY: setup-poetry
setup-poetry:
	@# Install poetry
	@curl -sSL https://install.python-poetry.org | python3 -
	@export PATH=$PATH:$HOME/.poetry/bin
	@poetry --version
	@poetry env info
	@poetry config virtualenvs.in-project true
	@poetry config repositories.testpypi https://test.pypi.org/legacy/
	@poetry env info
	@poetry config --list
	@rm -rf .venv
	@rm -f poetry.lock

.PHONY: poetry-windows
poetry-windows:
	@# Install poetry (Windows)
	@# system Python should be installed in advance
	@(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
	@export PATH=$PATH:$HOME/.poetry/bin
	@poetry --version
	@poetry config virtualenvs.in-project true
	@poetry env info
	@poetry config --list
	@rm -rf .venv
	@rm -f poetry.lock

.PHONY: install
install:
	@python -m pip install --upgrade pip
	@poetry self update
	@poetry install --with test,docs

.PHONY: update
update:
	@python -m pip install --upgrade pip
	@poetry self update
	@poetry update
	@poetry install --with test,docs

.PHONY: add
add:
	@# for main dependencies: make add pandas
	@python -m pip install --upgrade pip
	@poetry self update
	@poetry add ${target}@latest
	@poetry install --with test,docs

.PHONY: add-dev
add-dev:
	@# for test dependencies: make add target=pytest group=test
	@# for docs dependencies: make add target=Sphinx group=docs
	@python -m pip install --upgrade pip
	@poetry self update
	@poetry add ${target}@latest --group ${group}
	@poetry install --with test,docs

.PHONY: remove
remove:
	@# for main dependencies: make remove pandas
	@python -m pip install --upgrade pip
	@poetry self update
	@poetry remove ${target}
	@poetry install --with test,docs

.PHONY: remove-dev
remove-dev:
	@# for test dependencies: make remove target=pytest group=test
	@# for docs dependencies: make remove target=Sphinx group=docs
	@python -m pip install --upgrade pip
	@poetry self update
	@poetry remove ${target} --group ${group}
	@poetry install --with test,docs

.PHONY: check
check:
	@# Check codes with deptry and pflake8
	@poetry run deptry .
	@poetry run pflake8 covsirphy

.PHONY: test
test:
	@# All tests: make test
	@# Selected tests: make test target=/test_scenario.py::TestScenario
	@make check
	@poetry run pytest tests${target}

.PHONY: docs
docs:
	# docs/index.rst must be updated to include the notebooks
	@cp --force example/*.ipynb docs/
	@sudo apt install pandoc -y
	# Save markdown files in docs directory
	# docs/markdown/*md will be automatically included
	@cp --force .github/CODE_OF_CONDUCT.md docs/CODE_OF_CONDUCT.md
	@cp --force .github/CONTRIBUTING.md docs/CONTRIBUTING.md
	@cp --force SECURITY.md docs/SECURITY.md
	@pandoc -f commonmark -o docs/README.rst README.md --to rst
	# Create API reference
	@poetry run sphinx-apidoc -o docs covsirphy -fMT
	# Execute sphinx
	@cd docs; poetry run make html; cp -a _build/html/. ../docs

.PHONY: pypi
pypi:
	@# https://pypi.org/
	@# poetry config pypi-token.pypi "API Token of PyPi"
	@rm -rf covsirphy.egg-info/*
	@rm -rf dist/*
	@poetry publish --build

.PHONY: test-pypi
test-pypi:
	@# poetry config pypi-token.testpypi "API Token of Test-PyPi"
	@poetry config repositories.testpypi https://test.pypi.org/legacy/
	@rm -rf covsirphy.egg-info/*
	@rm -rf dist/*
	@poetry publish -r testpypi --build

.PHONY: clean
clean:
	@rm -rf input
	@rm -rf prof
	@rm -rf .pytest_cache
	@find -name __pycache__ | xargs --no-run-if-empty rm -r
	@rm -rf dist covsirphy.egg-info
	@rm -f .coverage*
	@poetry cache clear . --all
