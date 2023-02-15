# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module contains the fairness objectives. These functions won't be called as is,
as these will be passed to a `IndicesInput.objective`.
"""
import numpy as np

from deel.fairsense.utils.dataclasses import IndicesInput


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
    return np.float32(self.model(x if x is not None else self.x))


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
