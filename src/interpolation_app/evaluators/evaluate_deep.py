from pathlib import Path
from interpolation_app.interpolation.deep_simple import DeepCNNInterpolator
from interpolation_app.evaluation_runner import run_evaluation

if __name__ == "__main__":
    checkpoint = "checkpoints/simple-cnn-epoch=29-val_loss=0.1278.ckpt"
    interpolator = DeepCNNInterpolator(checkpoint_path=checkpoint)
    run_evaluation(interpolator, Path("results/deep_simple"))