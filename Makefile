PYTHON ?= python

.PHONY: all test check train capacity optimizers baselines clean

all: test check train capacity optimizers baselines

test:
	$(PYTHON) -m pytest

check:
	$(PYTHON) -m experiments.gradient_check

train:
	$(PYTHON) -m experiments.train_mnist

capacity:
	$(PYTHON) -m experiments.capacity

optimizers:
	$(PYTHON) -m experiments.optimizers

# needs the predictions that train saves
baselines: train
	$(PYTHON) -m experiments.baselines

clean:
	rm -rf figures results .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
