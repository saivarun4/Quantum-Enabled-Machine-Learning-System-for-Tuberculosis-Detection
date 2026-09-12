# TB Chest X-Ray Predictor

## Overview

TB Chest X-Ray Predictor is a software application for classifying chest X-ray images as **Normal** or **Tuberculosis**. It combines a FastAPI service, a pre-trained quantum machine learning model, and a browser-based interface for interactive analysis.

The application is intended for research and decision support. It is not a medical diagnosis and must not replace review by a qualified radiologist or clinician.

## Capabilities

- Upload a chest X-ray through the browser interface or REST API.
- Classify an image as Normal or Tuberculosis and return a confidence score.
- Generate explainable AI visualizations using occlusion mapping and quantum-model sensitivity.
- Create an HTML and PDF medical imaging report with patient and clinical metadata.
- Record prediction metadata in a local JSONL log.
- Optionally persist report artifacts through IBM Cloud Object Storage.
- Protect prediction endpoints with an optional API key.
- Enforce a configurable maximum upload size.

## Application Architecture

1. An image is uploaded through the web interface or an API request.
2. The image is converted to grayscale, resized to 16 x 16 pixels, and normalized.
3. The pre-trained model reduces the image features with PCA and evaluates them with a quantum circuit.
4. The service returns the classification, confidence, report identifier, and storage status.
5. Report generation adds model explanations and creates downloadable HTML and PDF outputs.

## Requirements

- Python 3.12 or later
- The dependencies listed in `requirements.txt`
- The included `qml_tb_classifier_model.pkl` model file

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application locally:

```bash
uvicorn app:app --reload --port 7860
```

Open `http://127.0.0.1:7860/` in a browser to use the application.

## API Reference

### Health check

```http
GET /health
```

### Classify an image

```bash
curl -X POST http://127.0.0.1:7860/predict \
  -F "file=@path/to/chest-xray.png"
```

The response includes `prediction`, `confidence`, `label`, `report_id`, and `storage` fields.

### Generate a report

Send a multipart `POST` request to `/generate-report` with these fields:

- `file`: the chest X-ray image
- `name`: patient name
- `age`: patient age
- `gender`: patient gender
- `patient_id`: patient identifier
- `referrer`: referring physician
- `clinical_notes`: clinical notes

The response includes the prediction, explainable visualizations, HTML report content, and a Base64-encoded PDF report.

Interactive API documentation is available at `http://127.0.0.1:7860/docs` while the application is running.


## Project Structure

```text
app.py                         FastAPI application and web interface
predict.py                     Command-line prediction utility
qml_tb_classifier.py           Data preparation and quantum model logic
qml_tb_classifier_model.pkl    Pre-trained model artifact
requirements.txt               Python dependencies
tests/                         Automated tests
TB_Chest_Radiography_Database/ Training and evaluation images
```

## Testing

Run the test suite from the project root:

```bash
python -m pytest -q
```

The tests cover image preprocessing, dataset loading, model persistence, and runtime port configuration.

## Responsible Use

This application is a machine learning research system. Predictions may be affected by image quality, dataset limitations, demographic differences, and clinical context. Always treat results as decision support and require qualified clinical review before making medical decisions.
