from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pennylane as qml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_ROOT = Path(__file__).resolve().parent / "TB_Chest_Radiography_Database"
IMAGE_SIZE = (16, 16)


def preprocess_image(image_path: str | os.PathLike[str], image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1).astype(np.float32)


def load_dataset(dataset_root: str | os.PathLike[str] | None = None, max_samples_per_class: int = 200, image_size: Tuple[int, int] = IMAGE_SIZE):
    root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
    normal_dir = root / "Normal"
    tb_dir = root / "Tuberculosis"

    if not normal_dir.exists() or not tb_dir.exists():
        raise FileNotFoundError(f"Expected folders {normal_dir} and {tb_dir} under {root}")

    normal_files = sorted(normal_dir.glob("*.png"))[:max_samples_per_class]
    tb_files = sorted(tb_dir.glob("*.png"))[:max_samples_per_class]

    if not normal_files or not tb_files:
        raise ValueError("No image files found in the dataset folders")

    X = []
    y = []

    for image_path in normal_files:
        X.append(preprocess_image(image_path, image_size=image_size))
        y.append(0)

    for image_path in tb_files:
        X.append(preprocess_image(image_path, image_size=image_size))
        y.append(1)

    X_arr = np.stack(X, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    return X_arr, y_arr


def build_quantum_model(n_qubits: int = 4, n_layers: int = 2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_qubits)]))

    def predict(features, weights):
        return np.array([circuit(feature, weights) for feature in features])

    return circuit, predict


def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def reduce_features(X: np.ndarray, n_qubits: int = 4, pca: PCA | None = None):
    if X.shape[1] < n_qubits:
        raise ValueError("Feature count must be at least the number of qubits")
    if pca is None:
        pca = PCA(n_components=n_qubits)
        transformed = pca.fit_transform(X)
    else:
        transformed = pca.transform(X)
    return transformed, pca


def save_model(model_payload: dict, output_path: str | os.PathLike[str] | None = None) -> Path:
    target_path = Path(output_path) if output_path is not None else Path(__file__).resolve().parent / "qml_tb_classifier_model.pkl"
    with target_path.open("wb") as handle:
        pickle.dump(model_payload, handle)
    return target_path


def load_model(model_path: str | os.PathLike[str] | None = None):
    target_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "qml_tb_classifier_model.pkl"
    with target_path.open("rb") as handle:
        return pickle.load(handle)


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


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int = 4,
    epochs: int = 40,
    learning_rate: float = 0.1,
):
    X_train, X_test, y_train, y_test, _ = prepare_data(X, y)
    X_train, pca = reduce_features(X_train, n_qubits=n_qubits)
    X_test, _ = reduce_features(X_test, n_qubits=n_qubits, pca=pca)

    circuit, predict = build_quantum_model(n_qubits=n_qubits)

    weights = np.zeros((2, n_qubits, 3), dtype=np.float64)
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for _ in range(epochs):
        def loss_fn(w):
            preds = predict(X_train, w)
            probs = sigmoid(preds)
            return -np.mean(y_train.astype(np.float64) * np.log(probs + 1e-8) + (1.0 - y_train.astype(np.float64)) * np.log(1.0 - probs + 1e-8))

        weights, _ = opt.step_and_cost(loss_fn, weights)

    train_probs = sigmoid(predict(X_train, weights))
    test_probs = sigmoid(predict(X_test, weights))

    train_pred_labels = (train_probs >= 0.5).astype(int)
    test_pred_labels = (test_probs >= 0.5).astype(int)

    train_acc = float(np.mean(train_pred_labels == y_train))
    test_acc = float(np.mean(test_pred_labels == y_test))
    return {
        "weights": weights,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "n_qubits": n_qubits,
        "train_predictions": train_pred_labels,
        "test_predictions": test_pred_labels,
        "pca": pca,
    }


def main():
    parser = argparse.ArgumentParser(description="Train or predict with the tuberculosis QML classifier")
    parser.add_argument("--image", type=str, help="Path to a single image for prediction")
    parser.add_argument("--model", type=str, default=str(Path(__file__).resolve().parent / "qml_tb_classifier_model.pkl"), help="Path to the saved model pickle")
    parser.add_argument("--train", action="store_true", help="Train a new model and save it")
    args = parser.parse_args()

    if args.image:
        label, score = predict_single_image(args.image, args.model)
        label_name = "Tuberculosis" if label == 1 else "Normal"
        print(f"Prediction: {label_name} (confidence={score:.3f})")
        return

    if args.train:
        X, y = load_dataset(max_samples_per_class=80)
        result = train_model(X, y, n_qubits=4, epochs=20)
        print("Train accuracy:", round(result["train_accuracy"], 4))
        print("Test accuracy:", round(result["test_accuracy"], 4))

        model_path = save_model(result, args.model)
        print("Model saved to:", model_path)
    else:
        print("No image provided. Use --image <path> for prediction or --train to train a model.")


if __name__ == "__main__":
    main()
