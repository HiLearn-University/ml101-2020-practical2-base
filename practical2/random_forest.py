import numpy as np


class RandomForestClassifier(object):
    def __init__(self):
        """
        you can add as many parameters as you want to your classifier
        """
        pass

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        :param data: array of features for each point
        :param labels: array of labels for each point
        """
        raise NotImplementedError()

    def predict(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


def f1_score(y_true: np.ndarray, y_predicted: np.ndarray):
    """
    only 0 and 1 should be accepted labels and 1 is the possitive class
    """
    assert set(y_true).union({1, 0}) == {1, 0}
    raise NotImplementedError()

