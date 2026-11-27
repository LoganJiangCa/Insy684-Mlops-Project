# insy684-mlops-project

## Model API (Dario)

The API loads a version of `digits-decision-tree` from MLflow, serves
`POST /predict`, saves predictions to PostgreSQL, and exposes Prometheus
metrics at `/metrics`. See [API setup and integration contract](docs/model-api.md)
for input format, database columns, Docker instructions, and tests.

`compose.api-test.yml` is a disposable API acceptance stack. It does not replace
the team's final Compose stack, training flow, frontend, or monitoring services.

## Planned Architecture

```mermaid
flowchart LR
    User[User] -->|Enters data| Frontend[Web frontend]
    Frontend -->|Sends input| API[Model API]
    API -->|Returns prediction| Frontend
    Frontend -->|Displays result| User
    Data[Training data] -->|Feeds| Train[Prefect training flow]
    Train -->|Logs parameters and metrics| Tracking[MLflow Tracking]
    Train -->|Registers selected model| Registry[MLflow Model Registry]
    API -->|Loads registered model| Registry
```

## Planned Monitoring

```mermaid
flowchart TD
    Sim[Traffic simulator] -->|Sends normal or drifted inputs| API[Model API]
    API -->|Logs inputs and predictions| DB[PostgreSQL]
    Sim -->|Stores known labels for test requests| DB
    Evidently -->|Reads logged data and writes drift and performance metrics| DB
    Prom[Prometheus] -->|Scrapes request count, latency and errors| API
    Grafana[Grafana dashboards and alerts] -->|Queries API metrics| Prom
    Grafana -->|Queries drift and model performance| DB
```
