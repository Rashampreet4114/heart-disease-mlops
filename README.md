# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end MLOps pipeline for the UCI Heart Disease dataset: EDA → feature engineering →
model training with MLflow tracking → FastAPI serving → Docker → Kubernetes → monitoring.

## Demo video

<video src="https://github.com/Rashampreet4114/heart-disease-mlops/raw/main/demo/heart_disease_pipeline_demo.mp4" controls width="700">
</video>

If the video doesn't render above, [watch/download it directly](demo/heart_disease_pipeline_demo.mp4).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Download raw dataset
python scripts/download_data.py

# 2. Clean data (binary target, impute missing values)
python -m src.data_processing

# 3. Train + tune Logistic Regression, Random Forest, XGBoost (MLflow-tracked)
python -m src.train

# View MLflow UI
mlflow ui   # http://127.0.0.1:5000

# 4. Lint + test
flake8 src/ api/ tests/ scripts/
pytest tests/ -v --html=screenshots/pytest_report.html --self-contained-html
```

## Run the API locally (no Docker)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Predict: `POST http://localhost:8000/predict` (see `api/schemas.py` for the request body)
- Metrics: http://localhost:8000/metrics
- Dashboard: http://localhost:8000/dashboard

## Run with Docker

```bash
docker build -t heart-disease-api:latest .
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
curl http://localhost:8000/health
```

## Deploy to Kubernetes (Docker Desktop Kubernetes or Minikube)

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
```

The service is of type `LoadBalancer`. On Docker Desktop's local Kubernetes, the assigned
external IP is not directly routable from the host, so access it via `kubectl port-forward`:

```bash
kubectl port-forward svc/heart-disease-api-service 8080:80
curl http://localhost:8080/health
```

On a real cloud cluster (EKS/GKE/AKS) the `EXTERNAL-IP` shown by `kubectl get svc` would be
reachable directly.

## Project structure

```
data/            raw + cleaned dataset
scripts/         download_data.py, build_eda_notebook.py
notebooks/       01_eda.ipynb
src/             data_processing.py, pipeline.py, train.py
api/             FastAPI app (main.py, schemas.py)
models/          saved model.joblib + metadata.json
tests/           pytest suite
k8s/             deployment.yaml, service.yaml
.github/workflows/ci.yml   lint -> test -> train CI pipeline
screenshots/     EDA plots, confusion matrices, ROC curves, deployment evidence
demo/            pipeline walkthrough video
report/          final report (report.docx), architecture diagram, video script
```
