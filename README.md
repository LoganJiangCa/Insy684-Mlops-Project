# insy684-mlops-project
## Planned Architecture

```mermaid
flowchart LR
    User[User] -->|Enters data| Frontend[Web frontend]
    Frontend -->|Sends input| API[Model API]
    API -->|Returns prediction| Frontend
    Frontend -->|Displays result| User
```
