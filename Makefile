.PHONY: poetry-linux
poetry-linux:
	@# Install poetry (Linux, OSX, WSL)
	@# system Python should be installed in advance
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
	@export PATH=$PATH:$HOME/.poetry/bin
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
	@pip install --upgrade pip
	@poetry self update
	@poetry update

.PHONY: add
add:
	@pip install --upgrade pip
	@poetry self update
	@poetry add ${target}

.PHONY: add-dev
add-dev:
	@pip install --upgrade pip
	@poetry self update
	@poetry add ${target} --dev

.PHONY: remove
remove:
	@pip install --upgrade pip
	@poetry self update
	@poetry remove ${target}

.PHONY: remove-dev
remove-dev:
	@pip install --upgrade pip
	@poetry self update
	@poetry remove ${target} --dev

.PHONY: test
test:
	@# All tests: make test
	@# Selected tests: make test target=/test_scenario.py::TestScenario
	@poetry run flake8 covsirphy --ignore=E501
	@poetry run pytest tests${target} -v --durations=0 --failed-first --maxfail=1 \
	 --cov=covsirphy --cov-report=term-missing

.PHONY: flake8
flake8:
	@poetry run flake8 covsirphy --ignore=E501

# https://github.com/sphinx-doc/sphinx/issues/3382
.PHONY: docs
docs:
	# docs/index.rst must be updated to include the notebooks
	@cp --no-clobber example/usage_*.ipynb docs/
	# Save markdown files in docs directory
	# docs/markdown/*md will be automatically included
	@cp --no-clobber .github/CODE_OF_CONDUCT.md docs/CODE_OF_CONDUCT.md
	@cp --no-clobber .github/CONTRIBUTING.md docs/CONTRIBUTING.md
	@cp --no-clobber SECURITY.md docs/SECURITY.md
	# Convert README.md to README.rst
	@# sudo apt install pandoc
	@pandoc --from markdown --to rst README.md -o docs/README.rst
	# Create API reference
	@poetry run sphinx-apidoc -o docs covsirphy -fMT
	# Execute sphinx
	@cd docs; poetry run make html; cp -a _build/html/. ../docs

.PHONY: pypi
pypi:
	@# poetry config http-basic.pypi <username> <password>
	@rm -rf covsirphy.egg-info/*
	@rm -rf dist/*
	@poetry publish --build

.PHONY: test-pypi
test-pypi:
	@# poetry config http-basic.testpypi <username> <password>
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
	@poetry cache clear . --all
	@pip install --upgrade pip
	@poetry self update
	@poetry update

.PHONY: setup-anyenv
setup-anyenv:
	@ # Set-up anyenv in Bash (Linux, WSL)
	@git clone https://github.com/riywo/anyenv ~/.anyenv
	@echo 'export PATH="$HOME/.anyenv/bin:$PATH"' >> ~/.bashrc; source ~/.bashrc
	@anyenv install --init: echo 'eval "$(anyenv init -)"' >> ~/.bashrc; source ~/.bashrc
	@/bin/mkdir -p $(anyenv root)/plugins; git clone https://github.com/znz/anyenv-update.git $(anyenv root)/plugins/anyenv-update
	@ # set-up pyenv
	@anyenv install pyenv;  exec $SHELL -l

.PHONY: setup-latest-python
setup-latest-python:
	@# Install the latest stable version of Pythonand set default for CovsirPhy project
	@anyenv update --force
	@version=`pyenv install -l | grep -x '  [0-9]\.[0-9]\.[0-9]' | tail -n 1 | tr -d ' '`; echo python $version
	@pyenv install $version; pyenv local $version; anyenv versions
	@# Install dependencies
	@rm -rf .venv
	@rm -f poetry.lock
	@pip install --upgrade pip
	@poetry install
	@poetry env info
