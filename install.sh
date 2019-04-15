#!/bin/bash

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
