import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from qml_tb_classifier import load_model, reduce_features, build_quantum_model

app = FastAPI(title="TB Chest X-Ray QML API")

MODEL_PATH = Path(__file__).resolve().parent / "qml_tb_classifier_model.pkl"
LOG_PATH = Path(__file__).resolve().parent / "prediction_log.jsonl"


def get_runtime_port() -> int:
    port = os.getenv("PORT", "7860")
    try:
        return int(port)
    except (TypeError, ValueError):
        return 7860


def append_prediction_log(filename: str, label: str, confidence: float, label_id: int) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "filename": filename,
        "prediction": label,
        "confidence": round(confidence, 3),
        "label": label_id,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = image.resize((16, 16))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok", "message": "TB chest X-ray classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    features = preprocess_uploaded_image(contents)

    model = load_model(MODEL_PATH)
    n_qubits = int(model.get("n_qubits", 4))
    weights = model.get("weights")
    pca = model.get("pca")

    if weights is None or pca is None:
        return JSONResponse(status_code=500, content={"error": "Model is missing weights or PCA reducer"})

    features_reduced, _ = reduce_features(features, n_qubits=n_qubits, pca=pca)
    _, predict = build_quantum_model(n_qubits=n_qubits)

    probs = 1.0 / (1.0 + np.exp(-predict(features_reduced, weights)))
    score = float(probs[0])
    label = int(score >= 0.5)
    label_name = "Tuberculosis" if label == 1 else "Normal"

    append_prediction_log(file.filename or "unknown", label_name, score, label)

    return JSONResponse(content={
        "prediction": label_name,
        "confidence": round(score, 3),
        "label": label,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=get_runtime_port())
