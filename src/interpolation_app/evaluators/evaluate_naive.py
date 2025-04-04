from pathlib import Path
from interpolation_app.interpolation.naive import NaiveInterpolator
from interpolation_app.evaluation_runner import run_evaluation


if __name__ == "__main__":
    interpolator = NaiveInterpolator()
    run_evaluation(interpolator, Path("results/naive"))
