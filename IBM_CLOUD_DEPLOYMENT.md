# IBM Cloud Deployment

This service can run on IBM Cloud Code Engine using the existing Dockerfile. The trained model is included in the image; the dataset and local virtual environment are excluded.

## Required services

- IBM Cloud Code Engine
- IBM Cloud Container Registry
- IBM Cloud Object Storage with a private bucket

## Configure secrets

Create an IBM COS service credential with access to the report bucket. Set the values from `.env.example` as Code Engine secrets or environment variables. The application requires `APP_API_KEY` in production. The browser sends this value as `X-API-Key` from the access-key field.

The application enforces a 10 MB upload limit by default. Set `MAX_UPLOAD_BYTES` lower or higher as appropriate for the deployment.

## Build and deploy

```powershell
ibmcloud login
ibmcloud plugin install container-registry
ibmcloud plugin install code-engine
ibmcloud cr login
ibmcloud cr namespace-add tb-qml

docker build -t us.icr.io/tb-qml/tb-qml-api:1.0 .
docker push us.icr.io/tb-qml/tb-qml-api:1.0

ibmcloud ce project create --name tb-qml-project
ibmcloud ce project select --name tb-qml-project
ibmcloud ce secret create --name tb-qml-config `
  --from-literal APP_API_KEY=$env:APP_API_KEY `
  --from-literal IBM_COS_BUCKET=$env:IBM_COS_BUCKET `
  --from-literal IBM_COS_ENDPOINT=$env:IBM_COS_ENDPOINT `
  --from-literal IBM_COS_API_KEY=$env:IBM_COS_API_KEY `
  --from-literal IBM_COS_SERVICE_INSTANCE_ID=$env:IBM_COS_SERVICE_INSTANCE_ID `
  --from-literal IBM_COS_AUTH_ENDPOINT=$env:IBM_COS_AUTH_ENDPOINT `
  --from-literal MAX_UPLOAD_BYTES=10485760

ibmcloud ce application create --name tb-qml-api `
  --image us.icr.io/tb-qml/tb-qml-api:1.0 `
  --port 8000 --cpu 1 --memory 2G `
  --min-scale 0 --max-scale 1 `
  --env-from-secret tb-qml-config

ibmcloud ce application get --name tb-qml-api
```

Use the HTTPS URL returned by Code Engine. Do not put IBM credentials in the Docker image or source code.

## Persistence behavior

For each report, the application writes:

- `reports/<report-id>.pdf` to IBM COS
- `logs/<report-id>.json` to IBM COS

The local JSONL log remains only as a development fallback. If COS settings are missing, `storage` is empty in the API response and no persistent cloud copy is created.

## Production notes

- Keep the COS bucket private and restrict the service credential to the required bucket.
- Rotate `APP_API_KEY` and COS credentials regularly.
- Put an identity-aware proxy or IBM API management layer in front of the service for multi-user clinical deployments.
- Do not use this application as an autonomous diagnosis system. Reports state that results require qualified clinical review.