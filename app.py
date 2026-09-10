import base64
import io
import json
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import ibm_boto3
from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ibm_botocore.client import Config
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from qml_tb_classifier import build_quantum_model, load_model, reduce_features

app = FastAPI(title="TB Chest X-Ray QML API")

MODEL_PATH = Path(__file__).resolve().parent / "qml_tb_classifier_model.pkl"
LOG_PATH = Path(__file__).resolve().parent / "prediction_log.jsonl"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
APP_API_KEY = os.getenv("APP_API_KEY", "")
COS_BUCKET = os.getenv("IBM_COS_BUCKET", "")
COS_ENDPOINT = os.getenv("IBM_COS_ENDPOINT", "")
COS_API_KEY = os.getenv("IBM_COS_API_KEY", "")
COS_SERVICE_INSTANCE_ID = os.getenv("IBM_COS_SERVICE_INSTANCE_ID", "")
COS_AUTH_ENDPOINT = os.getenv("IBM_COS_AUTH_ENDPOINT", "https://iam.cloud.ibm.com/identity/token")

MEDICAL_DISCLAIMER = "This AI result is decision support only. It is not a diagnosis and must be reviewed by a qualified radiologist or clinician."


def require_api_key(api_key: str | None) -> None:
  if APP_API_KEY and not api_key:
    raise PermissionError("An API key is required")
  if APP_API_KEY and not secrets.compare_digest(api_key, APP_API_KEY):
    raise PermissionError("Invalid API key")


def get_cos_client():
  if not all((COS_BUCKET, COS_ENDPOINT, COS_API_KEY, COS_SERVICE_INSTANCE_ID)):
    return None
  return ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY,
    ibm_service_instance_id=COS_SERVICE_INSTANCE_ID,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
  )


def persist_cos_object(key: str, body: bytes, content_type: str) -> bool:
  client = get_cos_client()
  if client is None:
    return False
  client.put_object(Bucket=COS_BUCKET, Key=key, Body=body, ContentType=content_type)
  return True


def persist_prediction_artifacts(report_id: str, log_entry: dict, report_pdf: bytes | None = None) -> dict[str, str]:
  storage: dict[str, str] = {}
  log_key = f"logs/{report_id}.json"
  if persist_cos_object(log_key, json.dumps(log_entry).encode("utf-8"), "application/json"):
    storage["log_key"] = log_key
  if report_pdf is not None:
    report_key = f"reports/{report_id}.pdf"
    if persist_cos_object(report_key, report_pdf, "application/pdf"):
      storage["report_key"] = report_key
  return storage


@app.middleware("http")
async def enforce_request_size(request: Request, call_next):
  content_length = request.headers.get("content-length")
  if content_length and int(content_length) > MAX_UPLOAD_BYTES:
    limit_mb = round(MAX_UPLOAD_BYTES / (1024 * 1024), 2)
    return JSONResponse(status_code=413, content={"error": f"Request exceeds the {limit_mb} MB limit"})
  return await call_next(request)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TB Chest X-Ray Predictor</title>
    <style>
      :root {
        --bg: #061322;
        --panel: rgba(15, 23, 42, 0.88);
        --primary: #38bdf8;
        --accent: #2dd4bf;
        --danger: #f87171;
        --success: #4ade80;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --shadow: 0 25px 80px rgba(5, 120, 170, 0.35);
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: Inter, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%), radial-gradient(circle at bottom right, rgba(45, 212, 191, 0.15), transparent 25%), var(--bg);
        color: var(--text);
        padding: 32px 18px;
      }

      .container {
        width: min(1200px, 100%);
        margin: 0 auto;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 30px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
        overflow: hidden;
      }

      .header {
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.28), rgba(14, 116, 144, 0.64));
        padding: 28px 30px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      }

      .header h1 {
        margin: 0;
        font-size: clamp(2.1rem, 4vw, 3.2rem);
        letter-spacing: -0.04em;
      }

      .header p {
        margin: 10px 0 0;
        color: var(--muted);
        font-size: 1rem;
      }

      .content {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 24px;
        padding: 30px;
      }

      .panel {
        background: rgba(15, 23, 42, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 24px;
        padding: 22px;
      }

      .form-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
      }

      .field {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .field label {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--muted);
      }

      .field input, .field select, .field textarea {
        width: 100%;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 12px 14px;
        color: var(--text);
        font: inherit;
      }

      .field textarea {
        min-height: 100px;
        resize: vertical;
      }

      .disclaimer {
        margin: 18px 30px 0;
        padding: 12px 14px;
        border-left: 3px solid #f59e0b;
        background: rgba(120, 53, 15, 0.22);
        color: #fde68a;
        font-size: 0.86rem;
      }

      .upload-zone {
        margin-top: 18px;
        min-height: 250px;
        border: 2px dashed rgba(56, 189, 248, 0.7);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(45, 212, 191, 0.04));
        transition: 0.2s ease;
      }

      .upload-zone.dragover {
        border-color: var(--accent);
        background: rgba(45, 212, 191, 0.14);
        transform: translateY(-2px);
      }

      .upload-inner {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 18px;
        text-align: center;
        padding: 20px;
      }

      .upload-icon {
        width: 68px;
        height: 68px;
        border-radius: 18px;
        display: grid;
        place-items: center;
        font-size: 2rem;
        background: rgba(56, 189, 248, 0.12);
      }

      .upload-inner h2 {
        margin: 0;
        font-size: 1.5rem;
      }

      .upload-inner p {
        margin: 0;
        color: var(--muted);
      }

      .file-input, .hidden { display: none; }

      .browse-btn, .submit-btn, .download-btn {
        appearance: none;
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }

      .browse-btn {
        background: linear-gradient(135deg, var(--primary), #2563eb);
        padding: 12px 18px;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.35);
      }

      .submit-btn {
        width: 100%;
        margin-top: 18px;
        background: linear-gradient(135deg, var(--accent), #0ea5e9);
        padding: 14px 18px;
        box-shadow: 0 14px 32px rgba(14, 165, 233, 0.35);
      }

      .download-btn {
        background: linear-gradient(135deg, #8b5cf6, #a78bfa);
        padding: 12px 16px;
        margin-top: 16px;
      }

      .download-pdf-btn {
        background: linear-gradient(135deg, #f97316, #ef4444);
        margin-left: 8px;
      }

      .browse-btn:hover, .submit-btn:hover, .download-btn:hover {
        transform: translateY(-1px);
      }

      .image-preview {
        width: 100%;
        min-height: 290px;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.9));
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .image-preview img {
        width: 100%;
        height: 100%;
        max-height: 420px;
        object-fit: cover;
        display: none;
      }

      .placeholder {
        color: var(--muted);
        text-align: center;
        padding: 25px;
      }

      .result-box {
        margin-top: 22px;
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: rgba(15, 23, 42, 0.95);
      }

      .result-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        background: rgba(148, 163, 184, 0.14);
        color: var(--text);
      }

      .result-box h3 {
        margin: 12px 0 8px;
        font-size: 1.1rem;
      }

      .score {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
      }

      .status-normal { background: rgba(74, 222, 128, 0.15); color: #bbf7d0; }
      .status-tuberculosis { background: rgba(248, 113, 113, 0.12); color: #fecaca; }

      .report-panel {
        margin-top: 18px;
        display: none;
      }

      .report-content {
        background: #f8fafc;
        color: #0f172a;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.18);
      }

      .report-header {
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        color: white;
        padding: 26px 26px 18px;
      }

      .report-header h2 {
        margin: 0;
        font-size: 2rem;
      }

      .report-body {
        padding: 28px 26px 26px;
      }

      .patient-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px 18px;
        margin-bottom: 22px;
      }

      .info-box {
        background: #eef2ff;
        border-radius: 12px;
        padding: 12px 14px;
        border: 1px solid #dbeafe;
      }

      .info-box small {
        display: block;
        color: #475569;
        margin-bottom: 4px;
      }

      .report-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        align-items: start;
        margin-top: 20px;
      }

      .report-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px;
      }

      .report-card img {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #dbeafe;
        display: block;
        margin-top: 10px;
      }

      .report-card h4 {
        margin: 0;
        font-size: 1.05rem;
      }

      .report-card p {
        margin: 8px 0 0;
        color: #475569;
      }

      @media (max-width: 820px) {
        .content, .form-grid, .patient-grid, .report-row { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>TB Chest X-Ray Predictor</h1>
        <p>Upload a patient chest radiograph to estimate whether it appears Normal or consistent with Tuberculosis.</p>
      </div>
      <div class="disclaimer">This AI result is decision support only. It is not a diagnosis and must be reviewed by a qualified radiologist or clinician.</div>

      <div class="content">
        <div class="panel">
          <form id="uploadForm">
            <div class="form-grid">
              <div class="field">
                <label for="patientName">Patient name</label>
                <input id="patientName" name="patientName" type="text" placeholder="e.g. John Smith" required />
              </div>
              <div class="field">
                <label for="patientAge">Age</label>
                <input id="patientAge" name="patientAge" type="number" min="0" placeholder="e.g. 42" required />
              </div>
              <div class="field">
                <label for="patientGender">Gender</label>
                <select id="patientGender" name="patientGender">
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div class="field">
                <label for="patientId">Patient ID</label>
                <input id="patientId" name="patientId" type="text" placeholder="e.g. PT-1042" />
              </div>
              <div class="field" style="grid-column: 1 / -1;">
                <label for="referrer">Referring physician</label>
                <input id="referrer" name="referrer" type="text" placeholder="e.g. Dr. Rao" />
              </div>
              <div class="field" style="grid-column: 1 / -1;">
                <label for="clinicalNotes">Clinical notes</label>
                <textarea id="clinicalNotes" name="clinicalNotes" placeholder="Short symptoms or clinical context for the report..."></textarea>
              </div>
              <div class="field" style="grid-column: 1 / -1;">
                <label for="accessKey">Deployment access key</label>
                <input id="accessKey" name="accessKey" type="password" autocomplete="off" placeholder="Required when authentication is enabled" />
              </div>
            </div>

            <div id="uploadZone" class="upload-zone">
              <div class="upload-inner">
                <div class="upload-icon">🩺</div>
                <h2>Choose chest X-ray image</h2>
                <p>PNG, JPG, or JPEG</p>
                <input id="fileInput" class="file-input" type="file" accept="image/*" />
                <button type="button" class="browse-btn" id="browseButton">Browse files</button>
              </div>
            </div>

            <button type="submit" class="submit-btn">Analyze and generate report</button>
          </form>
        </div>

        <div class="panel">
          <div class="image-preview" id="imagePreview">
            <div class="placeholder">Image preview will appear here</div>
            <img id="previewImage" alt="Uploaded chest X-ray preview" />
          </div>

          <div class="result-box" id="resultBox">
            <div class="result-label" id="resultBadge">Awaiting image</div>
            <h3>Prediction</h3>
            <p class="score" id="predictionText">--</p>
            <p id="confidenceText" style="color: var(--muted); margin: 6px 0 0;">Confidence: --</p>
          </div>

          <div id="reportPanel" class="report-panel">
            <div id="reportPreview" class="report-content"></div>
            <button type="button" id="downloadReportBtn" class="download-btn">Download medical report</button>
            <button type="button" id="downloadPdfBtn" class="download-btn download-pdf-btn">Download PDF</button>
          </div>
        </div>
      </div>
    </div>

    <script>
      const fileInput = document.getElementById('fileInput');
      const browseButton = document.getElementById('browseButton');
      const uploadForm = document.getElementById('uploadForm');
      const uploadZone = document.getElementById('uploadZone');
      const previewImage = document.getElementById('previewImage');
      const resultBadge = document.getElementById('resultBadge');
      const predictionText = document.getElementById('predictionText');
      const confidenceText = document.getElementById('confidenceText');
      const reportPanel = document.getElementById('reportPanel');
      const reportPreview = document.getElementById('reportPreview');
      const downloadReportBtn = document.getElementById('downloadReportBtn');
      const downloadPdfBtn = document.getElementById('downloadPdfBtn');

      browseButton.addEventListener('click', () => fileInput.click());

      fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (!file) return;
        const objectUrl = URL.createObjectURL(file);
        previewImage.src = objectUrl;
        previewImage.style.display = 'block';
        document.querySelector('.placeholder').style.display = 'none';
      });

      ['dragenter', 'dragover'].forEach((eventName) => {
        uploadZone.addEventListener(eventName, (event) => {
          event.preventDefault();
          uploadZone.classList.add('dragover');
        });
      });

      ['dragleave', 'drop'].forEach((eventName) => {
        uploadZone.addEventListener(eventName, (event) => {
          event.preventDefault();
          uploadZone.classList.remove('dragover');
        });
      });

      uploadZone.addEventListener('drop', (event) => {
        const file = event.dataTransfer.files[0];
        if (!file) return;
        fileInput.files = event.dataTransfer.files;
        const objectUrl = URL.createObjectURL(file);
        previewImage.src = objectUrl;
        previewImage.style.display = 'block';
        document.querySelector('.placeholder').style.display = 'none';
      });

      uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
          alert('Please choose a chest X-ray image first.');
          return;
        }

        resultBadge.textContent = 'Analyzing...';
        resultBadge.className = 'result-label';
        predictionText.textContent = 'Processing...';
        confidenceText.textContent = 'Machine learning analysis is running...';
        reportPanel.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('name', document.getElementById('patientName').value);
        formData.append('age', document.getElementById('patientAge').value);
        formData.append('gender', document.getElementById('patientGender').value);
        formData.append('patient_id', document.getElementById('patientId').value);
        formData.append('referrer', document.getElementById('referrer').value);
        formData.append('clinical_notes', document.getElementById('clinicalNotes').value);

        try {
          const accessKey = document.getElementById('accessKey').value;
          const response = await fetch('/generate-report', {
            method: 'POST',
            headers: accessKey ? { 'X-API-Key': accessKey } : {},
            body: formData,
          });

          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || data.error || 'Prediction failed');
          }

          const label = data.prediction;
          const confidence = Number(data.confidence || 0);
          const normalizedConfidence = Math.round(confidence * 1000) / 10;

          resultBadge.textContent = label;
          resultBadge.classList.remove('status-normal', 'status-tuberculosis');
          resultBadge.classList.add(label === 'Tuberculosis' ? 'status-tuberculosis' : 'status-normal');

          predictionText.textContent = label;
          confidenceText.textContent = `Confidence: ${normalizedConfidence}%`;
          reportPreview.innerHTML = data.report_html;
          reportPanel.style.display = 'block';

          downloadReportBtn.onclick = () => {
            const htmlBlob = new Blob([`<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><style>body{font-family:Arial,sans-serif;margin:24px;color:#0f172a}</style></head><body>${data.report_html}</body></html>`], { type: 'text/html' });
            const url = URL.createObjectURL(htmlBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${(data.patient_name || 'patient').replace(/\\s+/g, '-').toLowerCase()}-tb-report.html`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          };

          downloadPdfBtn.onclick = () => {
            const binary = atob(data.pdf_base64);
            const bytes = new Uint8Array(binary.length);
            for (let index = 0; index < binary.length; index += 1) {
              bytes[index] = binary.charCodeAt(index);
            }
            const pdfBlob = new Blob([bytes], { type: 'application/pdf' });
            const url = URL.createObjectURL(pdfBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${(data.patient_name || 'patient').replace(/\\s+/g, '-').toLowerCase()}-tb-report.pdf`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          };
        } catch (error) {
          resultBadge.textContent = 'Error';
          resultBadge.className = 'result-label';
          predictionText.textContent = 'Unable to classify image';
          confidenceText.textContent = error.message || 'Please try another image.';
        }
      });
    </script>
  </body>
</html>
"""


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def append_prediction_log(filename: str, label: str, confidence: float, label_id: int, report_id: str) -> dict:
    entry = {
    "report_id": report_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "filename": filename,
        "prediction": label,
        "confidence": round(confidence, 3),
        "label": label_id,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = image.resize((16, 16))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


def predict_probability(features: np.ndarray, model: dict) -> tuple[float, int]:
    n_qubits = int(model.get("n_qubits", 4))
    weights = model.get("weights")
    pca = model.get("pca")
    if weights is None or pca is None:
        raise ValueError("Model is missing weights or PCA reducer")

    features_reduced, _ = reduce_features(features, n_qubits=n_qubits, pca=pca)
    _, predict_fn = build_quantum_model(n_qubits=n_qubits)
    probs = sigmoid(predict_fn(features_reduced, weights))
    score = float(probs[0])
    label = int(score >= 0.5)
    return score, label


def _image_to_base64(array: np.ndarray, title: str, cmap: str = "gray") -> str:
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor="#f8fafc")
    ax.imshow(array, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _heatmap_to_base64(array: np.ndarray, title: str, cmap: str = "RdBu_r", signed: bool = False) -> str:
    arr = np.asarray(array, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor="#f8fafc")
    if signed:
        limit = float(np.max(np.abs(arr)))
        normalized = np.zeros_like(arr) if limit == 0 else arr / limit
        image = ax.imshow(normalized, cmap=cmap, vmin=-1.0, vmax=1.0)
    else:
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        normalized = np.zeros_like(arr) if arr_max == arr_min else (arr - arr_min) / (arr_max - arr_min)
        image = ax.imshow(normalized, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _overlay_to_base64(image_array: np.ndarray, heatmap: np.ndarray, title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor="#f8fafc")
    ax.imshow(image_array, cmap="gray", vmin=0.0, vmax=1.0)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap_limit = float(np.max(np.abs(heatmap)))
    if heatmap_limit == 0:
        normalized = np.zeros_like(heatmap)
    else:
      normalized = heatmap / heatmap_limit
    image = ax.imshow(normalized, cmap="RdBu_r", vmin=-1.0, vmax=1.0, alpha=0.65)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _quantum_feature_chart_to_base64(contributions: np.ndarray, title: str) -> str:
    values = np.asarray(contributions, dtype=np.float64)
    labels = [f"Q{i + 1}" for i in range(values.size)]
    colors_for_bars = ["#dc2626" if value >= 0 else "#2563eb" for value in values]
    fig, ax = plt.subplots(figsize=(5.2, 3.1), facecolor="#f8fafc")
    positions = np.arange(values.size)
    ax.bar(positions, values, color=colors_for_bars, width=0.58)
    ax.axhline(0.0, color="#334155", linewidth=0.8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Local contribution to TB score")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.01, 0.03, "Red: raises TB score | Blue: lowers TB score", transform=ax.transAxes, fontsize=8, color="#475569")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_xai_visuals(image_bytes: bytes, model: dict) -> dict[str, str]:
    source_image = Image.open(io.BytesIO(image_bytes)).convert("L")
    display_image = np.asarray(source_image, dtype=np.float32) / 255.0
    model_image = source_image.resize((16, 16), Image.Resampling.LANCZOS)
    model_array = np.asarray(model_image, dtype=np.float32) / 255.0
    features = model_array.reshape(1, -1)
    score, _ = predict_probability(features, model)

    n_qubits = int(model.get("n_qubits", 4))
    weights = model.get("weights")
    pca = model.get("pca")
    features_reduced, _ = reduce_features(features, n_qubits=n_qubits, pca=pca)
    circuit, _ = build_quantum_model(n_qubits=n_qubits)
    baseline_reduced = features_reduced[0].astype(np.float64)
    quantum_inputs = qml.numpy.array(baseline_reduced, requires_grad=True)
    quantum_weights = qml.numpy.array(weights, requires_grad=False)
    expectation_gradient = np.asarray(qml.grad(circuit, argnums=0)(quantum_inputs, quantum_weights), dtype=np.float64)
    sensitivity = score * (1.0 - score) * expectation_gradient
    contributions = sensitivity * baseline_reduced
    grad_map = pca.inverse_transform(sensitivity.reshape(1, -1)).reshape(16, 16).astype(np.float32)

    occlusion = np.zeros((16, 16), dtype=np.float32)
    step = 4
    for row in range(0, 16, step):
        for col in range(0, 16, step):
            masked = model_array.copy()
            masked[row:row + step, col:col + step] = np.mean(masked)
            masked_score, _ = predict_probability(masked.reshape(1, -1), model)
            occlusion[row:row + step, col:col + step] = score - masked_score

    target_size = source_image.size
    occlusion_display = np.asarray(
        Image.fromarray(occlusion).resize(target_size, Image.Resampling.NEAREST),
        dtype=np.float32,
    )
    sensitivity_display = np.asarray(
        Image.fromarray(grad_map).resize(target_size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    return {
        "original": _image_to_base64(display_image, "Original image"),
        "occlusion": _heatmap_to_base64(occlusion_display, "Occlusion effect: red raises TB score, blue lowers it", signed=True),
        "qml_sensitivity": _overlay_to_base64(display_image, sensitivity_display, "QML parameter-shift sensitivity: red raises TB score, blue lowers it"),
        "quantum_features": _quantum_feature_chart_to_base64(contributions, "Quantum-feature contribution chart"),
    }


def build_medical_report_html(patient_name: str, age: str, gender: str, patient_id: str, referrer: str, clinical_notes: str, prediction: str, confidence: float, xai: dict) -> str:
    conf_pct = round(confidence * 100, 2)
    return f"""
    <div class="report-content">
      <div class="report-header">
        <h2>Medical Imaging Report</h2>
        <div style="margin-top:12px; color:#dbeafe;">TB Chest X-ray Interpretation</div>
      </div>
      <div class="report-body">
        <div class="patient-grid">
          <div class="info-box"><small>Patient name</small><strong>{patient_name or 'Not provided'}</strong></div>
          <div class="info-box"><small>Age</small><strong>{age or 'N/A'}</strong></div>
          <div class="info-box"><small>Gender</small><strong>{gender or 'Not specified'}</strong></div>
          <div class="info-box"><small>Patient ID</small><strong>{patient_id or 'N/A'}</strong></div>
          <div class="info-box" style="grid-column: 1 / -1;"><small>Referring physician</small><strong>{referrer or 'Not provided'}</strong></div>
          <div class="info-box" style="grid-column: 1 / -1;"><small>Clinical notes</small><strong>{clinical_notes or 'No notes provided'}</strong></div>
        </div>

        <div class="report-row">
          <div class="report-card">
            <h4>Chest X-ray image</h4>
            <img src="data:image/png;base64,{xai['original']}" alt="Original X-ray image" />
          </div>
          <div class="report-card">
            <h4>Prediction summary</h4>
            <p><strong>Diagnosis:</strong> {prediction}</p>
            <p><strong>Confidence:</strong> {conf_pct}%</p>
            <p><strong>Assessment:</strong> {'This scan is consistent with a tuberculosis-like pattern.' if prediction == 'Tuberculosis' else 'This scan appears within the normal range for the model.'}</p>
          </div>
        </div>

        <div class="report-row">
          <div class="report-card">
            <h4>Explainable AI: Occlusion mapping</h4>
            <img src="data:image/png;base64,{xai['occlusion']}" alt="Occlusion mapping explanation" />
            <p>Red regions are areas where masking lowers the tuberculosis score, while blue regions are areas where masking raises it. Stronger colors indicate a larger score change.</p>
          </div>
          <div class="report-card">
            <h4>Explainable AI: QML parameter-shift sensitivity</h4>
            <img src="data:image/png;base64,{xai['qml_sensitivity']}" alt="Quantum parameter-shift sensitivity explanation" />
            <p>This map uses the quantum circuit's parameter-shift gradient with respect to the four PCA-reduced inputs. The gradient is converted through the classifier sigmoid and projected back to image space. Red raises the tuberculosis score; blue lowers it.</p>
          </div>
        </div>

        <div class="report-card" style="margin-top:20px;">
          <h4>Explainable AI: Quantum-feature contribution</h4>
          <img src="data:image/png;base64,{xai['quantum_features']}" alt="Quantum feature contribution chart" />
          <p>Q1 to Q4 are the four PCA-reduced values encoded into the quantum circuit. Each bar is the local contribution, calculated as quantum parameter-shift sensitivity multiplied by the corresponding reduced input. Red raises the tuberculosis score; blue lowers it.</p>
        </div>

        <div class="report-card" style="margin-top:20px;">
          <h4>Interpretive note</h4>
          <p>{MEDICAL_DISCLAIMER}</p>
        </div>
      </div>
    </div>
    """


def _reportlab_image(encoded_image: str, width: float = 3.15 * inch) -> ReportImage:
    image_bytes = io.BytesIO(base64.b64decode(encoded_image))
    image = ReportImage(image_bytes, width=width, height=width * 0.82)
    image.hAlign = "CENTER"
    return image


def build_medical_report_pdf(
    patient_name: str,
    age: str,
    gender: str,
    patient_id: str,
    referrer: str,
    clinical_notes: str,
    prediction: str,
    confidence: float,
    xai: dict[str, str],
) -> bytes:
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="TB Chest X-ray Medical Report",
        author="TB Chest X-ray Predictor",
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#1d4ed8")
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.leading = 14
    story = [
        Paragraph("TB Chest X-ray Medical Report", title_style),
        Paragraph("AI-assisted imaging assessment", styles["Normal"]),
        Spacer(1, 14),
    ]

    patient_data = [
        ["Patient name", escape(patient_name or "Not provided"), "Age", escape(age or "N/A")],
        ["Gender", escape(gender or "Not specified"), "Patient ID", escape(patient_id or "N/A")],
        ["Referring physician", escape(referrer or "Not provided"), "Report date", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    patient_table = Table(patient_data, colWidths=[1.35 * inch, 2.15 * inch, 1.35 * inch, 2.15 * inch])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0e7ff")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e0e7ff")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([patient_table, Spacer(1, 16)])

    confidence_percent = round(confidence * 100, 2)
    summary = Table([
        [Paragraph("Prediction", heading_style), Paragraph("Confidence", heading_style)],
        [Paragraph(escape(prediction), body_style), Paragraph(f"{confidence_percent}%", body_style)],
    ], colWidths=[3.5 * inch, 3.5 * inch])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([summary, Spacer(1, 14)])

    story.append(Paragraph("Clinical notes", heading_style))
    story.extend([Paragraph(escape(clinical_notes or "No notes provided"), body_style), Spacer(1, 14)])
    story.append(Paragraph("Images and explainable AI", heading_style))
    image_table = Table([
        [_reportlab_image(xai["original"]), _reportlab_image(xai["occlusion"])],
        [Paragraph("Original chest X-ray", body_style), Paragraph("Signed occlusion effect: red raises the TB score; blue lowers it", body_style)],
        [_reportlab_image(xai["qml_sensitivity"]), Paragraph("", body_style)],
        [Paragraph("QML parameter-shift sensitivity: red raises the TB score; blue lowers it", body_style), Paragraph("", body_style)],
        [_reportlab_image(xai["quantum_features"]), Paragraph("", body_style)],
        [Paragraph("Quantum-feature contribution: Q1-Q4 are PCA-reduced quantum inputs", body_style), Paragraph("", body_style)],
    ], colWidths=[3.5 * inch, 3.5 * inch])
    image_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, 1), 0.5, colors.HexColor("#cbd5e1")),
        ("GRID", (0, 2), (0, 5), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([image_table, Spacer(1, 14)])
    story.append(Paragraph(escape(MEDICAL_DISCLAIMER) + " The QML explanation uses a parameter-shift gradient through the quantum circuit and PCA back-projection; it is not a CNN Grad-CAM map.", body_style))
    document.build(story)
    return output.getvalue()


@app.get("/")
def home() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "message": "TB chest X-ray classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str | None = Header(default=None, alias="X-API-Key")) -> JSONResponse:
  try:
    require_api_key(api_key)
  except PermissionError as error:
    return JSONResponse(status_code=401, content={"error": str(error)})
    contents = await file.read()
  if len(contents) > MAX_UPLOAD_BYTES:
    return JSONResponse(status_code=413, content={"error": "Uploaded image exceeds the configured size limit"})
    model = load_model(MODEL_PATH)
    score, label = predict_probability(preprocess_uploaded_image(contents), model)
    label_name = "Tuberculosis" if label == 1 else "Normal"
  report_id = uuid.uuid4().hex
  log_entry = append_prediction_log(file.filename or "unknown", label_name, score, label, report_id)
  storage = persist_prediction_artifacts(report_id, log_entry)
  return JSONResponse(content={"prediction": label_name, "confidence": round(score, 3), "label": label, "report_id": report_id, "storage": storage})


@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    name: str = Form(""),
    age: str = Form(""),
    gender: str = Form(""),
    patient_id: str = Form(""),
    referrer: str = Form(""),
    clinical_notes: str = Form(""),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    try:
      require_api_key(api_key)
    except PermissionError as error:
      return JSONResponse(status_code=401, content={"error": str(error)})
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
      return JSONResponse(status_code=413, content={"error": "Uploaded image exceeds the configured size limit"})
    model = load_model(MODEL_PATH)
    score, label = predict_probability(preprocess_uploaded_image(contents), model)
    prediction = "Tuberculosis" if label == 1 else "Normal"
    xai = generate_xai_visuals(contents, model)
    report_html = build_medical_report_html(
        patient_name=name,
        age=age,
        gender=gender,
        patient_id=patient_id,
        referrer=referrer,
        clinical_notes=clinical_notes,
        prediction=prediction,
        confidence=score,
        xai=xai,
    )
    report_pdf = build_medical_report_pdf(
        patient_name=name,
        age=age,
        gender=gender,
        patient_id=patient_id,
        referrer=referrer,
        clinical_notes=clinical_notes,
        prediction=prediction,
        confidence=score,
        xai=xai,
    )
    report_id = uuid.uuid4().hex
    log_entry = append_prediction_log(file.filename or "unknown", prediction, score, label, report_id)
    storage = persist_prediction_artifacts(report_id, log_entry, report_pdf)
    return JSONResponse(content={
        "prediction": prediction,
        "confidence": round(score, 3),
        "patient_name": name,
      "report_id": report_id,
      "storage": storage,
        "report_html": report_html,
        "pdf_base64": base64.b64encode(report_pdf).decode("ascii"),
    })
