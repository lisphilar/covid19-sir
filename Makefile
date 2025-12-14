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

.PHONY: setup-uv
setup-uv:
	@# Install uv
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@export PATH="$$HOME/.local/bin:$$PATH"
	@uv --version
	@rm -rf .venv
	@rm -f uv.lock

.PHONY: install
install:
	@uv sync --group test --group docs

.PHONY: update
update:
	@uv sync --upgrade --group test --group docs

.PHONY: add
add:
	@# for main dependencies: make add target=pandas
	@uv add ${target}
	@uv sync --group test --group docs

.PHONY: add-dev
add-dev:
	@# for test dependencies: make add-dev target=pytest group=test
	@# for docs dependencies: make add-dev target=Sphinx group=docs
	@uv add ${target} --group ${group}
	@uv sync --group test --group docs

.PHONY: remove
remove:
	@# for main dependencies: make remove target=pandas
	@uv remove ${target}
	@uv sync --group test --group docs

.PHONY: remove-dev
remove-dev:
	@# for test dependencies: make remove-dev target=pytest group=test
	@# for docs dependencies: make remove-dev target=Sphinx group=docs
	@uv remove ${target} --group ${group}
	@uv sync --group test --group docs

.PHONY: check
check:
	@# Check codes with deptry and pflake8
	@uv run deptry .
	@uv run pflake8 covsirphy
	@uv run pyright

.PHONY: test
test:
	@# All tests: make test
	@# Selected tests: make test target=/test_scenario.py::TestScenario
	@make check --no-print-directory
	@uv run pytest tests${target}

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
	@uv run sphinx-apidoc -o docs covsirphy -fMT -t=docs/_templates
	# Execute sphinx
	@cd docs; uv run make html --no-print-directory; cp -a _build/html/. ../docs

.PHONY: pypi
pypi:
	@# https://pypi.org/
	@# UV_PUBLISH_TOKEN="API Token of PyPi"
	@rm -rf dist/*
	@uv build
	@uv publish

.PHONY: test-pypi
test-pypi:
	@# UV_PUBLISH_TOKEN="API Token of Test-PyPi"
	@rm -rf dist/*
	@uv build
	@uv publish --publish-url https://test.pypi.org/legacy/

.PHONY: clean
clean:
	@rm -rf input
	@rm -rf prof
	@rm -rf .pytest_cache
	@find -name __pycache__ | xargs --no-run-if-empty rm -r
	@rm -rf dist covsirphy.egg-info
	@rm -f .coverage*
	@uv cache clean
	@rm importtime.log

.PHONY: shell
shell:
	@# To activate the virtual environment: source .venv/bin/activate
	@uv run python -i

.PHONY: importtime
importtime:
	@uv run python -X importtime -c "import covsirphy" 2> importtime.log
	@uv run tuna importtime.log

.PHONY: data
data:
	@uv run python ./data/vaccine_data.py

.PHONY: demo
demo:
	@uv run python ./example/demo.py
