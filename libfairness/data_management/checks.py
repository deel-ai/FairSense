import types
import pandas as pd
import numpy as np

def check_inputs_type(inputs):
    if inputs is not None and not check_is_array_or_df(inputs):
        raise TypeError(
            "FairnessProblem.inputs should be a numpy array or a pandas dataframe.")


def check_columns_type(columns):
    if columns is not None and not isinstance(columns, list):
        raise TypeError("FairnessProblem.columns should be a list.")


def check_outputs_type(outputs):
    if outputs is not None and not check_is_array_or_df(outputs):
        raise TypeError(
            "FairnessProblem.outputs should be a numpy array or a pandas dataframe.")


def check_function_type(function):
    if function is not None and not isinstance(function, types.FunctionType):
        raise TypeError("FairnessProblem.function should be a function.")


def check_labels_type(labels):
    if labels is not None and not check_is_array_or_df(labels):
        raise TypeError(
            "FairnessProblem.labels should be a numpy array or a pandas dataframe.")


def check_groups_studied_type(groups_studied):
    if not isinstance(groups_studied, list):
        raise TypeError("FairnessProblem.groups_studied should be a list.")
    if len(groups_studied) > 0:
        for elt in groups_studied:
            if not isinstance(elt, list):
                raise TypeError(
                    "FairnessProblem.groups_studied should be a list of lists.")


def check_categorical_features_type(categorical_features):
    if not isinstance(categorical_features, list):
        raise TypeError(
            "FairnessProblem.categorical_features should be a list.")


def check_is_array_or_df(x):
    return isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)
