import unittest

from data_validation import validate_training_data


class TestTrainingData(unittest.TestCase):
    def test_rejects_mismatched_sample_and_label_counts(self):
        features = [[10], [20], [30]]
        labels = [0, 1]

        with self.assertRaises(ValueError):
            validate_training_data(features, labels)
