---
title: TB Chest X-Ray Predictor
emoji: 🩺
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 8000
---

# TB Chest X-Ray QML API

This repository contains a FastAPI service that serves a pre-trained quantum machine learning model for tuberculosis classification on chest X-rays.

## Docker packaging

Build the Docker image:

```bash
docker build -t qml-tb-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 qml-tb-api
```

Verify the API is running:

```bash
curl http://localhost:8000/
```

## Docker Compose

Build and start the service with Compose:

```bash
docker compose up --build
```

## Hugging Face Spaces Deployment

Create a new Hugging Face Space with:

- **SDK:** Docker
- **Visibility:** Public or private, as appropriate

Push this repository to the Space repository. Hugging Face will build the existing
`Dockerfile` and expose the application on port `8000` using the `app_port` metadata
above.

In the Space **Settings**, add an optional secret named `APP_API_KEY`. When set,
the browser must send this value in the deployment access-key field. Do not commit
the key to the repository.

Hugging Face Space disk is temporary and may be reset when the Space restarts. The
application's local report and JSONL log fallback should therefore be treated as
temporary. IBM COS persistence requires the IBM COS environment variables described
in [IBM_CLOUD_DEPLOYMENT.md](IBM_CLOUD_DEPLOYMENT.md), but those credentials should
only be added as Space secrets.

After the build completes, open the Space URL and verify the health endpoint:

```text
https://YOUR-USERNAME-YOUR-SPACE.hf.space/
```

## Notes

- The Docker image includes the pre-trained `qml_tb_classifier_model.pkl`.
- Large development artifacts such as `TB_Chest_Radiography_Database/`, `.venv/`, and `tests/` are excluded from the image by `.dockerignore`.
- The API exposes port `8000`.

## Google Cloud Deployment

This repository includes a GitHub Actions workflow that can deploy the service to Google Cloud Run automatically.

### Required GitHub Secrets

- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GCP_REGION`: The Cloud Run region, e.g. `us-central1`
- `GCP_SERVICE_ACCOUNT_KEY`: JSON key for a service account with permissions:
  - Cloud Run Admin
  - Storage Admin (for Cloud Build)
  - Service Account User

### Deployment behavior

- The workflow runs on `push` to `main` and on manual dispatch
- It builds the Docker image, pushes it to Google Container Registry, and deploys to Cloud Run
- The deployed Cloud Run service is configured as public (`--allow-unauthenticated`)

### How to use

1. Create and configure a GitHub repository for this project.
2. Add the required secrets in GitHub repository settings.
3. Push the `main` branch.
4. View the workflow under GitHub Actions.
