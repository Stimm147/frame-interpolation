from pathlib import Path
from interpolation_app.interpolation.optical_flow import OpticalFlowInterpolator
from interpolation_app.evaluation_runner import run_evaluation


if __name__ == "__main__":
    interpolator = OpticalFlowInterpolator(alpha=0.5, use_multiscale=False)
    run_evaluation(interpolator, Path("results/optical_flow"))
