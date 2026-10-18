.PHONY: check test tables figures replay full
PYTHON ?= python3
check:
	$(PYTHON) reproduce.py check
test:
	$(PYTHON) -m pytest tests -q
tables:
	$(PYTHON) reproduce.py tables
figures:
	$(PYTHON) reproduce.py figures
replay:
	$(PYTHON) reproduce.py replay --all-cells --workers 4
full:
	$(PYTHON) reproduce.py simulate --reps 1000 --workers 8 --output generated/full
