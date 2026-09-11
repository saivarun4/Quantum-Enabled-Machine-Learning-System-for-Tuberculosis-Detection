# TB Chest X-Ray QML Classifier

This project classifies chest X-ray images as **Normal** or **Tuberculosis** using a small hybrid quantum machine learning (QML) model. It includes:

- A PennyLane quantum circuit used for binary classification.
- A training and command-line prediction workflow in `qml_tb_classifier.py`.
- A command-line prediction script in `predict.py`.
- A FastAPI service in `app.py` for image uploads.
- Docker and Docker Compose files for running the API.
- Unit tests for image preprocessing, dataset loading, model saving, and runtime configuration.

> This is an educational and research project. Its predictions are not a medical diagnosis and must not be used as a replacement for evaluation by a qualified healthcare professional.

## How the application works

The complete image-to-prediction flow is:

1. An image is opened with Pillow and converted to grayscale.
2. The image is resized to `16 x 16` pixels and normalized to values between `0` and `1`, producing `256` features.
3. During training, the features are standardized and reduced to four dimensions with PCA. Four dimensions are used because the default quantum circuit has four qubits.
4. The quantum circuit encodes the reduced features with `RY` rotations, applies trainable rotation gates and CNOT entanglement, and returns a Pauli-Z expectation value.
5. A sigmoid converts the circuit output into a score. Scores greater than or equal to `0.5` are labeled `Tuberculosis`; lower scores are labeled `Normal`.
6. Training stores the learned weights, fitted PCA reducer, qubit count, and accuracy values in `qml_tb_classifier_model.pkl`.
7. The API loads that model for each request, predicts the uploaded image, and appends the result to `prediction_log.jsonl`.

## Project structure

```text
app.py                         FastAPI health and prediction endpoints
predict.py                     Command-line prediction entry point
qml_tb_classifier.py           Dataset, preprocessing, QML model, and training logic
requirements.txt               Python dependencies
tests/                         Automated tests
TB_Chest_Radiography_Database/ Training images in Normal/ and Tuberculosis/
```

## Requirements

- Python 3.12 or a compatible Python version supported by the pinned dependencies.
- The chest X-ray dataset arranged as:

	```text
	TB_Chest_Radiography_Database/
	+-- Normal/
	+-- Tuberculosis/
	```

- Docker is optional and only required for containerized execution.

## Local setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train a model

The default training command reads up to 80 images from each dataset class, trains for 20 epochs, and writes the model to `qml_tb_classifier_model.pkl`:

```bash
python qml_tb_classifier.py --train
```

The command prints the training and test accuracy. A model file must exist before using either prediction interface.

## Predict from the command line

```bash
python predict.py path/to/chest_xray.png
```

Use a different saved model with:

```bash
python predict.py path/to/chest_xray.png --model path/to/model.pkl
```

## Run the API locally

Start the service with its default port:

```bash
python app.py
```

The API listens on port `7860` by default. Set `PORT` to use another port:

```powershell
$env:PORT = "8000"
python app.py
```

Check the health endpoint:

```text
GET http://localhost:7860/
```

Send an image to the prediction endpoint with `curl`:

```bash
curl -X POST -F "file=@path/to/chest_xray.png" http://localhost:7860/predict
```

Example response:

```json
{
	"prediction": "Normal",
	"confidence": 0.731,
	"label": 0
}
```

The service writes one JSON object per line to `prediction_log.jsonl`, including the timestamp, original filename, prediction, confidence, and numeric label.

## Docker

Build the image:

```bash
docker build -t qml-tb-api .
```

Run it with host port `8000` mapped to the application port `7860`:

```bash
docker run --rm -p 8000:7860 qml-tb-api
```

Verify the container:

```bash
curl http://localhost:8000/
```

The Docker image installs the pinned dependencies and copies the project, including the saved model when `qml_tb_classifier_model.pkl` is present in the build context. Large development artifacts should remain excluded by `.dockerignore` where applicable.

## Docker Compose

The existing Compose file can be started with:

```bash
docker compose up --build
```

Because the application listens on `7860` by default, use `http://localhost:8000` only when the Compose port mapping targets container port `7860`. Otherwise, set `PORT=8000` in the Compose environment and keep the `8000:8000` mapping.

## Testing

Run the automated tests from the project root:

```bash
python -m pytest
```

The tests use temporary images and directories, so they do not modify the training dataset.

## Contributing

Contributions are welcome. A useful contribution can improve preprocessing, model training, API behavior, documentation, testing, or deployment.

1. Create a focused branch for the change.
2. Install the project dependencies and run `python -m pytest` before making changes.
3. Keep changes small and explain the motivation in the pull request.
4. Add or update tests for changed behavior, especially preprocessing and prediction contracts.
5. Update this README when commands, file formats, endpoints, or configuration change.
6. Run the tests again and include relevant training or evaluation results for model changes.

When contributing model improvements, report the dataset split, number of samples per class, preprocessing settings, number of qubits, epochs, and both training and test accuracy. Avoid committing private patient data, generated prediction logs, virtual environments, or large model artifacts unless explicitly required.

## License and data

Check the dataset's own terms and attribution requirements before redistributing it. Do not use patient data outside the permissions and safeguards applicable to the dataset source.
