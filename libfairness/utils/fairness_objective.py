"""
This module contains the fairness objectives. These functions won't be called as is,
as these will be passed to a `IndicesInput.objective`.
"""
import numpy as np
from pandas import DataFrame
from libfairness.utils.dataclasses import IndicesInput


def y_true(self: IndicesInput, x=None):
    """
    Evaluate the intrinsic fairness of the dataset. This allow to check if the data
    used for training is biased.
    """
    if x is not None:
        raise RuntimeError("this target can only be used with x=None")
    return self.y_true


def y_pred(self: IndicesInput, x=None):
    """
    Evaluate the fairness of the model's predictions over the dataset. This allow to
    check if the model gives biased decisions.
    """
    return DataFrame(
        self.model(x if x is not None else self.x), dtype=float, columns=["outputs"]
    )


def classification_error(self: IndicesInput, x=None):
    """
    Evaluate the fairness of the model's errors over the dataset. This allow to
    check if the model errors are due to the presence of a sensitive attribute.

    The error is computed for classification by checking if the model output is equal to
    `y_true` given in the `IndicesInput`.
    """
    if x is not None:
        raise RuntimeError("this target can only be used with x=None")
    y_pred = self.model(self.x)
    # if len(y_pred.shape) < 2:
    #     y_pred = np.expand_dims(y_pred, -1)
    return np.not_equal(self.y_true, y_pred).astype(float)


def squared_error(self: IndicesInput, x=None):
    """
    Evaluate the fairness of the model's errors over the dataset. This allow to
    check if the model errors are due to the presence of a sensitive attribute.

    The error is computed for regression by measuring the squared error between the
    model output and `y_true` given in the `IndicesInput`.
    """
    if x is not None:
        raise RuntimeError("this target can only be used with x=None")
    y_pred = self.model(self.x)
    if len(y_pred.shape) < 2:
        y_pred = np.expand_dims(y_pred, -1)
    return np.square(self.y_true - y_pred)
