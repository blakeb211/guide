# Makefile for running iterate.py

.PHONY: run

run:
	python3 iterate.py

test:
	cd tests && pytest -rA --log-cli-level INFO test_datasets.py

test_unbiased:
	cd tests && pytest -rA --log-cli-level INFO -vv --tb=long test_unbiased.py
