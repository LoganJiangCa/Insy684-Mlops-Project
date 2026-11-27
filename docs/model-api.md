# Model API

Loads `digits-decision-tree` from MLflow and saves successful predictions to PostgreSQL.

## Endpoints

- `POST /predict`: predict one digit.
- `GET /health`: check the model and database.
- `GET /metrics`: Prometheus request count, latency, and errors.
- `GET /docs`: try the API in a browser.

Send 64 numbers from an 8 × 8 image, flattened row by row. Training intensities are 0–16. `true_label` is optional and must be the known correct digit, not the prediction.

```json
{
  "features": [0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0,
               0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0,
               0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0,
               0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0],
  "true_label": 0
}
```

The response contains `request_id`, `prediction` (0–9), `model_name`, and `model_version`. Each successful request creates one record. Retrying creates another record.

Invalid inputs return 422, inference errors return 500, and database failures return 503. Finite inputs outside 0–16 are allowed for drift tests.

## Configuration

Copy `.env.example` to `.env`, set a local password, and keep `.env` out of Git.

| Variable | Value |
|---|---|
| `DATABASE_URL` | Required PostgreSQL URL; plain `postgresql://` also works |
| `MLFLOW_TRACKING_URI` | Shared registry; defaults to `sqlite:///mlflow.db` |
| `MODEL_NAME` | Defaults to `digits-decision-tree` |
| `MODEL_VERSION` | Numeric version; defaults to `1` |
| `CORS_ORIGINS` | Comma-separated frontend origins; empty by default |

The final Compose stack should build `model_api/Dockerfile` with target `runtime`, expose port 8000, and pass these variables. The registered model and its artifacts must be accessible from the container. Use the same model dependencies as training. Restart the API to change the deployed version.

## Prediction records and metrics

Table: `prediction_logs`, created by the API at startup.

| Column | Content |
|---|---|
| `request_id` | UUID returned to the caller |
| `created_at` | Insertion timestamp with time zone |
| `features` | JSONB array of 64 numbers |
| `prediction` | Predicted digit |
| `true_label` | Optional correct digit |
| `model_name`, `model_version` | Deployed model |
| `inference_ms` | Model inference time |

Asad can read this table for Evidently and update late labels by `request_id`. Exclude missing labels from accuracy calculations. Drift/performance results belong in separate tables.

Prometheus should scrape `api:8000/metrics`. Metrics are `http_requests_total`, `http_errors_total` (status ≥400), and `http_request_duration_seconds` (whole request). Filter `path="/predict"` for prediction traffic. Scrapes are excluded; unknown paths share `__unmatched__`. Counters reset on restart. Use one Uvicorn worker per container.

## Run the API test stack

Requires Docker with Compose and port 8000 available. This stack registers a separate test model using the notebook's dataset and tree settings. It does not run the final training flow or frontend.

```bash
export POSTGRES_PASSWORD=disposable-local-test-password
docker compose -p api-check -f compose.api-test.yml up -d --build --wait --wait-timeout 240 api
docker compose -p api-check -f compose.api-test.yml run --build --rm tests
```

Open `http://localhost:8000/docs` to try the request above. Clean up when finished:

```bash
docker compose -p api-check -f compose.api-test.yml down -v --remove-orphans
unset POSTGRES_PASSWORD
```

The GitHub `Model API` workflow runs the API tests and a real container integration test. To run component tests locally, install `requirements-api-test.txt`, set `TEST_DATABASE_URL` to a disposable PostgreSQL database, then run `python -m pytest tests/model_api/test_api.py -v`. These tests delete records from their test table.

Logan supplies the trained registry version; Antonin connects the frontend and final Compose stack; Asad connects the simulator and monitoring. A teammate must review the PR before merging with a merge commit or rebase, not squash.
