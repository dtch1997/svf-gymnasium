PYTHON?=python

.venv:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade setuptools pip
	.venv/bin/python -m pip install pip-tools
	touch .venv

.install_requires:
	.venv/bin/python -m pip install -r requirements/dev.in
	.venv/bin/python -m pip install -e third-party/sheeprl
	.venv/bin/python -m pip install -e .
	pre-commit install

test: 
	.venv/bin/python -m pytest -m pytest tests --cov=svf_gymnasium --cov-report=xml

install: .venv .install_requires

all: install test

.PHONY: .install_requires install test all