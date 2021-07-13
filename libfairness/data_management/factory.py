from libfairness.fairness_problem import FairnessProblem
import numpy as np
import pandas as pd
from .checks import *

def create_fairness_problem(inputs=None, columns=None, function=None, outputs=None, labels=None, groups_studied=[], categorical_features=[]):
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
    check_columns_type(columns)
    check_outputs_type(outputs)
    check_labels_type(labels)
    check_function_type(function)
    check_groups_studied_type(groups_studied)
    check_categorical_features_type(categorical_features)

    if columns == None:
        if isinstance(inputs, pd.DataFrame):
            columns = list(inputs.columns)
            inputs = inputs.to_numpy()
        else:
            columns = [str(i) for i in range(inputs.shape[1])]

    return FairnessProblem(inputs, columns, function, outputs, labels, groups_studied, categorical_features)
