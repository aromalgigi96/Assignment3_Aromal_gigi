# DEPLOYMENT.md

## Containerization and Deployment Steps

### 1. Dockerfile and .dockerignore

* **Dockerfile** uses `python:3.11-slim` base image.
* Installs dependencies from `requirements.txt` without cache.
* Copies application code at `/app` and exposes port `8080`.
* Runs FastAPI app via Uvicorn.

### 2. Build and Run Locally

```bash
# Build the Docker image
docker build -t penguin-api .

# Run the container mapping host port 8080 to container port 8080
docker run -d --name penguin-api -p 8080:8080 penguin-api
```

### 3. Testing Endpoints

```bash
# Health check
curl http://localhost:8080/health
# ⇒ {"status":"ok"}

# Root endpoint
curl http://localhost:8080/
# ⇒ {"message":"Hello! Welcome to the Penguins Classification API."}

# Prediction endpoint (PowerShell)
$body = '{"bill_length_mm":40.0,"bill_depth_mm":18.0,"flipper_length_mm":195,"body_mass_g":4000,"year":2008,"sex":"male","island":"Biscoe"}'
Invoke-RestMethod -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType 'application/json'
# ⇒ {"species":"Adelie"}
```

### 4. Issues Encountered and Solutions

* **Editable local path in requirements.txt**: Removed project path and retained only PyPI packages.
* **Python version mismatch for contourpy**: Upgraded base image to `python:3.11-slim` to satisfy dependencies.
* **PowerShell curl alias**: Used `Invoke-RestMethod` for POST requests.

### 5. Performance and Security Observations

* **Image Size**: 2.31 GB (optimized by slim image and `--no-cache-dir`).
* **Layers**: 17 layers (verified via `docker history`).
* **Security**: Running as root user (default) in this container; consider non-root user in production for least-privilege.

### 6. Summary of Image Layers & Size

* **Layer Count**: 17 layers
* **Image Size**: 2.31 GB


You’ve got it, Aromal! Here's that deployment documentation in clean and readable **Markdown format**—ready to drop into a `DEPLOYMENT.md` file or your project wiki:

```markdown
# 🐧 Penguin API Deployment Guide

## 7. Service Account & Environment Variables

### `.env` File (excluded from Git)
```env
GOOGLE_APPLICATION_CREDENTIALS=secrets/sa-key.json
GCS_BUCKET_NAME=penguin-model
GCS_BLOB_NAME=model.json
```

- Keep `sa-key.json` in the `secrets/` folder.
- Never commit secrets or `.env` to Git or Docker image.

---

## 8. Google Cloud Deployment (Manual from Windows Host)

### 8.1. Build & Tag Docker Image
```bash
docker build -t penguin-api .
docker tag penguin-api:latest us-central1-docker.pkg.dev/ml-deployment-demo-467801/penguin-api-repo/penguin-api:latest
```

### 8.2. Push Image to Artifact Registry
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/ml-deployment-demo-467801/penguin-api-repo/penguin-api:latest
```

### 8.3. Deploy to Cloud Run

- Go to **Cloud Run** > **Create Service**
- Select the container image from Artifact Registry:
  ```
  us-central1-docker.pkg.dev/ml-deployment-demo-467801/penguin-api-repo/penguin-api:latest
  ```
- Port: `8080`
- CPU/Memory: `1 CPU, 4 GB RAM`
- Environment Variables:
  - `GOOGLE_APPLICATION_CREDENTIALS=/gcp/sa-key.json`
  - `GCS_BUCKET_NAME=penguin-model`
  - `GCS_BLOB_NAME=model.json`
- Allow **unauthenticated invocations**

---

## 8.4. Run & Test

### Health Check
```bash
curl https://penguin-api-191199043596.us-central1.run.app/health
# Response: {"status":"ok"}
```

### Prediction Request
```bash
curl -X POST https://penguin-api-191199043596.us-central1.run.app/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"bill_length_mm\":40.0,\"bill_depth_mm\":18.0,\"flipper_length_mm\":195,\"body_mass_g\":4000,\"year\":2008,\"sex\":\"male\",\"island\":\"Biscoe\"}"
# Response: {"species":"Adelie"}
```

---

## 9. Deployment Issues & Fixes

- 🛠️ **Permissions/mount issues**: Fixed volume path and file existence
- 🕒 **Cold start delay**: ~10 seconds to load model from GCS
- 🔐 **Missing env vars**: Set correctly via Cloud Run UI

---

## 10. Performance

- Cold start: `< 10s`
- Warm prediction: `< 1s`
- Concurrency: Suitable for classroom load

---

## 11. Public URL

```text
https://penguin-api-191199043596.us-central1.run.app
```
```

Want help writing a `README.md` that links to this or maybe a diagram of your architecture next? Let’s jazz it up. 💡📦💻
