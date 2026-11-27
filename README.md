# insy684-mlops-project

## Model API (Dario)

Loads the registered digits model, serves `POST /predict`, saves predictions to
PostgreSQL, and exposes `/metrics`. See [setup](docs/model-api.md).

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

## Team Responsibilities

| Team member | Feature branch | Responsibilities |
| --- | --- | --- |
| Logan Jiang | `feature/mlflow-tracking` | Build the training pipeline with Prefect, track experiments in MLflow, and register the selected model. Maintain CI tests and automate Docker image publishing with GitHub Actions. |
| Dario Guo | `feature/model-api` | Build the FastAPI service that loads the registered model and returns predictions. Log predictions to PostgreSQL, expose API metrics, and test the API and database together. |
| Antonin Laferrière-Mailfait | `feature/prediction-frontend` | Build the web interface where users enter data and see predictions from the API. Set up Docker Compose so the team can run the application locally with one command. |
| Asad Nadeem | `feature/traffic-simulator` | Simulate normal traffic and changes in input data. Use Evidently to measure data drift and model performance, and build Grafana dashboards and alerts for model and API monitoring. |
