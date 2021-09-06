import types
import pandas as pd
import numpy as np


def check_inputs_type(inputs):
    """Check the type of the "inputs" attribute.

    Args:
        inputs (): inputs attribute for the fairness problem.

    Raises:
        TypeError: it isn't a numpy array or a pandas dataframe
    """
    if inputs is not None and not check_is_array_or_df(inputs):
        raise TypeError(
            "FairnessProblem.inputs should be a numpy array or a pandas dataframe."
        )


def check_columns_type(columns):
    """Check the type of the "columns" attribute.

    Args:
        columns (): columns attribute for the fairness problem.

    Raises:
        TypeError: it isn't a list
    """
    if columns is not None and not isinstance(columns, list):
        raise TypeError("FairnessProblem.columns should be a list.")


def check_outputs_type(outputs):
    """Check the type of the "outputs" attribute.

    Args:
        outputs (): outputs attribute for the fairness problem.

    Raises:
        TypeError: it isn't a numpy array or a pandas dataframe
    """
    if outputs is not None and not check_is_array_or_df(outputs):
        raise TypeError(
            "FairnessProblem.outputs should be a numpy array or a pandas dataframe."
        )


def check_function_type(function):
    """Check the type of the "function" attribute.

    Args:
        function (): function attribute for the fairness problem.

    Raises:
        TypeError: it isn't a function
    """
    if function is not None and not isinstance(function, types.FunctionType):
        raise TypeError("FairnessProblem.function should be a function.")


def check_labels_type(labels):
    """Check the type of the "labels" attribute.

    Args:
        labels (): labels attribute for the fairness problem.

    Raises:
        TypeError: it isn't a numpy array or a pandas dataframe
    """
    if labels is not None and not check_is_array_or_df(labels):
        raise TypeError(
            "FairnessProblem.labels should be a numpy array or a pandas dataframe."
        )


def check_groups_studied_type(groups_studied):
    """Check the type of the "groups_studied" attribute.

    Args:
        groups_studied (): groups_studied attribute for the fairness problem.

    Raises:
        TypeError: it isn't a list
        TypeError: it isn't a list of list
    """
    if not isinstance(groups_studied, list):
        raise TypeError("FairnessProblem.groups_studied should be a list.")
    if len(groups_studied) > 0:
        for elt in groups_studied:
            if not isinstance(elt, list):
                raise TypeError(
                    "FairnessProblem.groups_studied should be a list of lists."
                )


def check_categorical_features_type(categorical_features):
    """Check the type of the "categorical_features" attribute.

    Args:
        categorical_features (): categorical_features attribute for the fairness problem.

    Raises:
        TypeError: it isn't a list
    """
    if not isinstance(categorical_features, list):
        raise TypeError("FairnessProblem.categorical_features should be a list.")


def check_is_array_or_df(x):
    return isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)
