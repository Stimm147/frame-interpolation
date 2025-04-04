.PHONY: run test

run:
	set PYTHONPATH=src && poetry run python -m interpolation_app.evaluate_naive_interpolator

test:
	poetry run pytest tests

