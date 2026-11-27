FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY model_api ./model_api
EXPOSE 8000
CMD ["uvicorn", "model_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
COPY requirements-api-test.txt .
RUN pip install --no-cache-dir -r requirements-api-test.txt
COPY tests/model_api ./tests/model_api
CMD ["python", "-m", "pytest", "tests/model_api", "-q"]
