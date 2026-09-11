import argparse
import os
from pathlib import Path

import numpy as np

from qml_tb_classifier import load_model, preprocess_image, reduce_features, build_quantum_model


def predict_single_image(image_path: str | os.PathLike[str], model_path: str | os.PathLike[str] | None = None) -> tuple[int, float]:
    model = load_model(model_path)
    features = preprocess_image(image_path)
    features = features.reshape(1, -1)

    n_qubits = int(model.get("n_qubits", 4))
    weights = model.get("weights")
    if weights is None:
        raise ValueError("The saved model does not contain weights")

    pca = model.get("pca")
    if pca is None:
        raise ValueError("The saved model does not contain a fitted PCA reducer")

    features_reduced, _ = reduce_features(features, n_qubits=n_qubits, pca=pca)
    _, predict = build_quantum_model(n_qubits=n_qubits)

    probs = 1.0 / (1.0 + np.exp(-predict(features_reduced, weights)))
    score = float(probs[0])
    label = int(score >= 0.5)
    return label, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tuberculosis status from a chest X-ray image")
    parser.add_argument("image", type=str, help="Path to the input chest X-ray image")
    parser.add_argument("--model", type=str, default=str(Path(__file__).resolve().parent / "qml_tb_classifier_model.pkl"), help="Path to the saved model pickle")
    args = parser.parse_args()

    label, score = predict_single_image(args.image, args.model)
    label_name = "Tuberculosis" if label == 1 else "Normal"
    print(f"Prediction: {label_name} (confidence={score:.3f})")


if __name__ == "__main__":
    main()
