.PHONY: setup test lint format

setup:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	mypy qecml

format:
	ruff format .
