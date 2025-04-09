from interpolation_app.interpolation.morphing import MorphingInterpolator
from interpolation_app.evaluation_runner import run_evaluation
from pathlib import Path

if __name__ == "__main__":
    interpolator = MorphingInterpolator(alpha=0.5, method="SIFT")
    run_evaluation(interpolator, Path("results/morphing"))
