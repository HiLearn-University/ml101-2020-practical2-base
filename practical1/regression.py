import argparse
import numpy as np
from typing import List
import itertools
import matplotlib.pyplot as plt


def target(apartment: dict) -> float:
    """
    Return the variable we want to use to predict, the apartment price
    divided by 1000.
    We divide by 1000, to make numbers look less scary :)
    >>> target({'address': 'Yerevan', 'price': '102,000'})
    102.0
    """
    raise NotImplementedError()


def featurize(apartment: dict) -> np.ndarray:
    """
    :param apartment: a dictionary containing the data
    :return: feature vector for that point - np.ndarray
    """
    return np.array([
        1,
    ])


def fit_linear_regression(X: np.ndarray, Y: np.ndarray, *, l: float = 0) -> np.ndarray:
    """
    Fit linear regression to the data
    >>> fit_linear_regression(np.array([[1, 1], [1, 6]]), np.array([1, 2]))
    array([0.8, 0.2])
    """
    raise NotImplementedError()


def y_hat(x: np.ndarray, beta: np.ndarray) -> float or np.ndarray:
    """
    Note: This is different from homework 1.
    Here we are fitting linear regression! And we want to make y_hat to
    work both when x is a single point, and when x is a matrix!
    :param x: input vector or matrix
    :param beta: model parameters
    :return: prediction(s)
    >>> y_hat(np.array([1, 1]), np.array([0.1, 2.0]))
    2.1
    >>> y_hat(np.array([[1, 1], [2, 3]]), np.array([0.1, 2.0]))
    array([2.1, 6.2])
    """
    raise NotImplementedError()
