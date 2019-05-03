SHELL = /bin/bash
.PHONY: clean clean-test clean-pyc clean-build
.DEFAULT_GOAL := help

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

test: ## run tests quickly with the default Python
	py.test tests

install: # install virtualenv and requirements
	venv='.work'
	echo Install Virtualenv
	which -s virtualenv || pip install virtualenv
	# install python3
	which -s $venv/bin/python3 || virtualenv -p python3 $venv
	# source venv
	source $venv/bin/activate
	# Install requirements
	pip install -e git+https://github.com/openai/spinningup.git#egg=spinup
	pip3 install -e .
	pip3 install -r requirements.txt
	pip3 install pylint
