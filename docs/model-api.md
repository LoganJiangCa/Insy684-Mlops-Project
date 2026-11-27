# Model API: setup and team integration

## What this component does

The current notebook trains a decision tree on scikit-learn's **digits** dataset.
Each example is an 8 × 8 grayscale image of a handwritten digit. Flatten the
image **row by row** to obtain 64 numeric features, normally between 0 and 16.
The target and prediction are digit classes 0 through 9.

The API loads one registered MLflow version at startup. It does not retrain the
model or choose a new version on each request. A successful request runs that
model, saves a database record, then returns the prediction. Logging failure
returns HTTP 503 instead of claiming success.

## HTTP contract (frontend and simulator)

`POST /predict`, `Content-Type: application/json`:

```json
{
  "features": [0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0,
               0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0,
               0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0,
               0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0],
  "true_label": 0
}
```

`true_label` is optional. The simulator can provide the known correct label of
an unseen example, allowing Evidently to compare predictions against truth.
The frontend normally omits it. Do not use the predicted digit as the true label.

Response, HTTP 200 (request ID and predicted class vary):

```json
{
  "request_id": "7b7c11c3-9138-44bd-bb8b-e3591b5cc693",
  "prediction": 0,
  "model_name": "digits-decision-tree",
  "model_version": "1"
}
```

- Exactly 64 finite numbers are required; booleans and numeric strings are rejected.
- If supplied, `true_label` must be an integer from 0 to 9. Unknown fields are rejected.
- Invalid JSON inputs return 422. Finite values outside the training range are
  allowed so the simulator can test feature drift; they are not clipped.
- Inference errors return 500. Database connection/write failures return 503.
- Each successful call has a new UUID; retries are separate prediction records.
- `GET /health` checks startup-loaded model availability and database connectivity.
- `GET /docs` provides an interactive form for trying the endpoint.

The frontend must preprocess images to match **8 × 8 / intensity 0–16**, not
send a PNG file or a 28 × 28 MNIST vector. If another model is introduced, the
team must update and test this contract together.

## PostgreSQL contract (monitoring)

The API creates its own `prediction_logs` table if absent. It does not modify
other tables. Use a database role allowed to create this table on first startup.

| Column | PostgreSQL type | Meaning |
|---|---|---|
| `request_id` | UUID, primary key | Returned to the caller; join key for labels if needed |
| `created_at` | timestamptz | Time the record was inserted |
| `features` | jsonb | The 64 numbers in model input order |
| `prediction` | integer | Predicted class 0–9 |
| `true_label` | integer, nullable | Simulator's known correct class, if supplied |
| `model_name` | varchar(255) | Registered model name |
| `model_version` | varchar(64) | Pinned registered version |
| `inference_ms` | double precision | Model inference time only, milliseconds |

Only successful, persisted predictions are present. Validation, inference, and
database errors are counted in Prometheus instead. Missing true labels must be
excluded from accuracy calculations. For Evidently, expand `features` into 64
columns in the same order and compare them with training/reference data.
If labels arrive later, the simulator can update `true_label` by `request_id`
using its database connection. Evidently metrics belong in separate tables.

## Prometheus contract (monitoring)

Scrape `http://api:8000/metrics` inside the final Compose network:

```yaml
scrape_configs:
  - job_name: model-api
    static_configs:
      - targets: ["api:8000"]
```

- `http_requests_total{method,path,status}`: request count, including errors.
- `http_errors_total{method,path,status}`: responses with status 400 or higher.
- `http_request_duration_seconds{method,path}`: histogram of the whole request,
  including validation, inference, and database logging.

Filter `path="/predict"` when plotting prediction traffic. `/metrics` requests
are excluded to avoid counting scrapes. Unknown paths share `__unmatched__` to
avoid unbounded labels. These are process-local counters; restart resets them,
so use Prometheus `rate`/`increase`. Run **one Uvicorn worker per container**.
These operational metrics do not measure drift or model accuracy; those require
the simulator, labels, Evidently, and reference data.

## Configuration (training and Compose owners)

Copy `.env.example` to `.env` and replace the example password locally.
Never commit `.env`. URL-encode special characters in URL passwords.

| Variable | Required/default | Meaning |
|---|---|---|
| `DATABASE_URL` | Required | `postgresql+psycopg://user:password@postgres:5432/database` |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Same registry used by the training flow |
| `MODEL_NAME` | `digits-decision-tree` | Same name registered by training |
| `MODEL_VERSION` | `1` | Numeric registered version to deploy |
| `CORS_ORIGINS` | Empty | Allowed browser origins, comma separated |

Build from the repository root with Dockerfile `model_api/Dockerfile`, target
`runtime`. Set the environment variables above, expose port 8000, and connect
to the PostgreSQL and MLflow services on their shared network. Allow the exact
frontend origin in `CORS_ORIGINS` when browser requests cross origins.

MLflow must already contain the selected model version. Its artifacts must also
be accessible from the API container (for example, an MLflow server that serves
artifacts, or shared mounted storage at the same absolute paths). A copied
SQLite file alone is insufficient if model artifacts remain only on a laptop.
The current notebook's local registry is suitable for local experiments; Logan
and Antonin must connect the final training flow and API to the same registry
and accessible artifacts. Configure the desired version and restart the API to
deploy it. Missing model versions fail startup; no dummy model is substituted.

For local development, after PostgreSQL and the registered model exist:

```bash
python -m pip install -r requirements-api.txt
# Export DATABASE_URL and MLFLOW_TRACKING_URI for your own services first.
uvicorn model_api.app:create_app --factory --host 0.0.0.0 --port 8000
```

## Run the disposable container acceptance test

This creates a separate model fixture using the notebook's dataset, split and
tree parameters, registers it in a shared test registry, starts a PostgreSQL
container and API container, then calls the API over HTTP and checks the saved
row independently. It uses a held-out validation sample. It is **test support**,
not the final orchestrated training flow or a drift experiment.

Requirements: Docker with Compose, port 8000 available, and an internet connection
for the initial image/dependency downloads. From the repository root:

```bash
# Disposable test password only; no external account or cloud service is used.
export POSTGRES_PASSWORD=disposable-local-test-password
docker compose -p api-check -f compose.api-test.yml up -d --build --wait --wait-timeout 240 api
docker compose -p api-check -f compose.api-test.yml run --build --rm tests
```

A passing test confirms a real container prediction, its database record,
validation rejection and exported metrics. You can also open
`http://localhost:8000/docs` while the stack is running. Always clean up after:

```bash
docker compose -p api-check -f compose.api-test.yml down -v --remove-orphans
unset POSTGRES_PASSWORD
```

The test database container has no persistent data volume. `down -v` removes
the fixture model volume too. The commands use the isolated Compose project `api-check`; free/change port
8000 if another service already occupies it.

## Automated checks

The original lightweight CI and training validation tests remain intact. API
tests live in `tests/model_api/` and run in a separate workflow with their own
dependencies and disposable PostgreSQL service. Do not point test settings at a
database containing real records: fixtures delete rows in `prediction_logs`.

```bash
python -m pip install -r requirements-api-test.txt
# Set TEST_DATABASE_URL to a disposable PostgreSQL database first.
python -m pytest tests/model_api/test_api.py -v
python -m unittest discover -s tests -v
```

The `Model API` GitHub workflow also runs the Docker acceptance test on pushes
to this branch/main and on pull requests. Unit/component tests cover registered
model inference and persisted values, input dimensions/types, unknown labels,
drift inputs, database failure, inference failure, metrics, health, and missing
registry versions. The container test is skipped outside the Compose test stack.

## Remaining team integration

- Logan: register the trained model from the Prefect flow; keep compatible model
  dependencies; supply the registry URI, accessible artifacts, and version.
- Antonin: add this API image/service to the final single-command Compose stack,
  wire frontend preprocessing and requests, and configure its browser origin.
- Asad: send held-out/synthetic samples, preserve correct labels, read this
  table for Evidently, and add Prometheus/Grafana configuration and drift evidence.
- A teammate must review the API pull request. Merge with a merge commit or
  rebase, not squash, as required by the project guidelines. The team's overall
  publishing workflow, final stack, dashboards, and presentation remain group work.
