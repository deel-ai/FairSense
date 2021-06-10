from fairness_problem import FairnessProblem
import numpy as np
import pandas as pd
import types


def create_fairness_problem(inputs=None, function=None, outputs=None, labels=None, groups_studied=[]):
    """Create a fairness problem object.

    Args:
        inputs ([type], optional): [description]. Defaults to None.
        function ([type], optional): [description]. Defaults to None.
        outputs ([type], optional): [description]. Defaults to None.
        labels ([type], optional): [description]. Defaults to None.
        groups_studied ([type], optional): [description]. Defaults to None.

    Returns:
        FairnessProblem: An object representing a fairness problem.
    """
    check_inputs_type(inputs)
    check_outputs_type(outputs)
    check_labels_type(labels)
    check_function_type(function)
    check_groups_studied_type(groups_studied)

    return FairnessProblem(inputs, function, outputs, labels, groups_studied)

def check_inputs_type(inputs):
    if inputs is not None and not check_is_array_or_df(inputs):
        raise TypeError("FairnessProblem.inputs should be a numpy array or a pandas dataframe.")

def check_outputs_type(outputs):
    if outputs is not None and not check_is_array_or_df(outputs):
        raise TypeError("FairnessProblem.outputs should be a numpy array or a pandas dataframe.")

def check_function_type(function):
    if function is not None and not isinstance(function, types.FunctionType):
        raise TypeError("FairnessProblem.function should be a function.")

def check_labels_type(labels):
    if labels is not None and not check_is_array_or_df(labels):
        raise TypeError("FairnessProblem.labels should be a numpy array or a pandas dataframe.")

def check_groups_studied_type(groups_studied):
    if not isinstance(groups_studied, list):
        raise TypeError("FairnessProblem.groups_studied should be a list.")
    if len(groups_studied) > 0:
        for elt in groups_studied:
            if not isinstance(elt, list):
                raise TypeError("FairnessProblem.groups_studied should be a list of lists.")

def check_is_array_or_df(x):
    return isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)