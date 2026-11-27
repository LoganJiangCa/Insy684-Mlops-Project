def validate_training_data(features, labels):
    if len(features) != len(labels):
        raise ValueError("Each training sample must have one label.")
