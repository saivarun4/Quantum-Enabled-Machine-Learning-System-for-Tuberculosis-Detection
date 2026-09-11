import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from app import get_runtime_port
from qml_tb_classifier import load_dataset, preprocess_image, save_model


def test_preprocess_image_returns_normalized_vector():
    with TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "sample.png"
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(image_path)

        features = preprocess_image(image_path)

        assert features.shape == (256,)
        assert features.dtype == np.float32
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)


def test_load_dataset_returns_balanced_features_and_labels():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "Normal").mkdir(exist_ok=True)
        (root / "Tuberculosis").mkdir(exist_ok=True)

        for idx in range(2):
            Image.fromarray(np.full((16, 16), 10 + idx, dtype=np.uint8)).save(root / "Normal" / f"normal_{idx}.png")
            Image.fromarray(np.full((16, 16), 200 + idx, dtype=np.uint8)).save(root / "Tuberculosis" / f"tb_{idx}.png")

        X, y = load_dataset(root, max_samples_per_class=2, image_size=(16, 16))

        assert X.shape[0] == 4
        assert y.shape[0] == 4
        assert np.unique(y).tolist() == [0, 1]
        assert X.shape[1] == 256


def test_save_model_writes_pickle_file():
    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        payload = {"weights": np.array([0.1, 0.2]), "n_qubits": 2}

        save_model(payload, model_path)

        assert model_path.exists()
        with model_path.open("rb") as handle:
            loaded = pickle.load(handle)
        assert loaded["n_qubits"] == 2


def test_get_runtime_port_uses_space_default_and_env_override(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert get_runtime_port() == 7860

    monkeypatch.setenv("PORT", "9000")
    assert get_runtime_port() == 9000
