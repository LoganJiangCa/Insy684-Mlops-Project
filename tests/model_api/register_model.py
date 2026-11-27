"""Build a small *test fixture*, not a replacement for Logan's training flow."""

import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def register_fixture(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{directory / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "api-integration-fixture", artifact_location=str(directory / "artifacts")
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    features, labels = load_digits(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=67, stratify=labels
    )
    model = DecisionTreeClassifier(max_depth=8, random_state=67).fit(train_x, train_y)
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model,
            name="decision-tree",
            input_example=test_x[:1],
            registered_model_name="digits-decision-tree",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
    sample = {
        "features": test_x[0].tolist(),
        "true_label": int(test_y[0]),
        "expected_prediction": int(model.predict(test_x[:1])[0]),
    }
    (directory / "sample.json").write_text(json.dumps(sample))
    return sample


if __name__ == "__main__":
    directory = Path(os.environ.get("FIXTURE_DIR", "/mlflow"))
    if not (directory / "sample.json").exists():
        register_fixture(directory)
