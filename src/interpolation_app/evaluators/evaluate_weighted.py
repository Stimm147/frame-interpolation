from pathlib import Path
from interpolation_app.interpolation.weighted import WeightedAverageInterpolator
from interpolation_app.evaluation_runner import run_evaluation


if __name__ == "__main__":
    interpolator = WeightedAverageInterpolator(alpha=0.75)
    run_evaluation(interpolator, Path("results/weighted"))
