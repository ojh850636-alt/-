.PHONY: help deps test run docker-build docker-run clean

help:
	@echo "Available targets: deps test run docker-build docker-run clean"

deps:
	python -m pip install --upgrade pip
	if [ -f requirements-dev.txt ]; then python -m pip install -r requirements-dev.txt; fi
	if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi

run:
	python -m uvicorn lucia_ultimate_quantum_integrated_fixed:app --reload

test:
	python -m pytest -q

docker-build:
	docker build -t lucia-cleaned .

docker-run:
	docker run --rm -p 8000:8000 lucia-cleaned

clean:
	rm -rf __pycache__ .pytest_cache
