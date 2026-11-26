# insy684-mlops-project
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
