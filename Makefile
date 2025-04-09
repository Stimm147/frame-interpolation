.PHONY: run test

run-naive:
	set PYTHONPATH=src && poetry run python src/interpolation_app/evaluators/evaluate_naive.py

run-weighted:
	set PYTHONPATH=src && poetry run python src/interpolation_app/evaluators/evaluate_weighted.py

run-morphing:
	set PYTHONPATH=src && poetry run python src/interpolation_app/evaluators/evaluate_morphing.py

run-optical-flow:
	set PYTHONPATH=src && poetry run python src/interpolation_app/evaluators/evaluate_optical_flow.py

train-simple:
	set PYTHONPATH=src && poetry run python src/interpolation_app/trainers/simple_trainer.py

run-deep:
	set PYTHONPATH=src && poetry run python src/interpolation_app/evaluators/evaluate_deep.py

test:
	poetry run pytest tests

format:
	poetry run ruff format src tests