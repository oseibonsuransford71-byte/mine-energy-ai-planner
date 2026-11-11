.PHONY: install test run real-data

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

run:
	python scripts/run_pipeline.py

# rerun everything on real NASA POWER irradiance (needs internet)
real-data:
	python scripts/fetch_nasa_power.py
	python scripts/run_pipeline.py --weather data/raw/nasa_power_weather.csv
