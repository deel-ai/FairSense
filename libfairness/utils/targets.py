from typing import Callable
import numpy as np
from pandas import DataFrame

from libfairness.utils.dataclasses import IndicesInput


def y_true(self: IndicesInput, x=None):
    if x is not None:
        raise RuntimeError("this target can only be used with x=None")
    return self.y


def predictions(self: IndicesInput, x=None):
    return DataFrame(self.model(x if x is not None else self.x), columns=["outputs"])


def classification_error(self: IndicesInput, x=None):
    if x is not None:
        raise RuntimeError("this target can only be used with x=None")
    y_pred = self.model(self.x)
    if len(y_pred.shape) < 2:
        y_pred = np.expand_dims(y_pred, -1)
    return np.equal(self.y, y_pred)


def squared_error(self: IndicesInput, x=None):
    if x is not None:
        raise RuntimeError("this target can only be used with x=None")
    y_pred = self.model(self.x)
    if len(y_pred.shape) < 2:
        y_pred = np.expand_dims(y_pred, -1)
    return np.square(self.y - y_pred)
