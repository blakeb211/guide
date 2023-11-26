# Makefile for running iterate.py

.PHONY: run

run:
	python3 iterate.py

test:
	cd tests && pytest -rA --log-cli-level CRITICAL .
