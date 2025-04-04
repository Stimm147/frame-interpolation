from interpolation_app.utils.triplet_dataset import TripletDataset
from pathlib import Path
import numpy as np


def test_sample_dataset_loading():
    dataset_path = Path("data/sample_dataset")
    dataset = TripletDataset(dataset_path)

    assert len(dataset) > 0, "Sample dataset is empty or not loaded"

    sample = dataset[0]

    assert "before" in sample
    assert "after" in sample
    assert "ground_truth" in sample

    for key in ["before", "after", "ground_truth"]:
        image = sample[key]
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3  # height x width x channels
        assert image.shape[2] == 3  # RGB
        assert image.dtype == np.uint8
