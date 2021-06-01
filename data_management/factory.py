from fairness_problem import FairnessProblem
import numpy as np
import pandas as pd
import types


def create_fairness_problem(inputs=None, function=None, outputs=None, labels=None):
    check_inputs_type(inputs)
    check_outputs_type(outputs)
    check_labels_type(labels)
    check_function_type(function)

    return FairnessProblem(inputs, function, outputs, labels)

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

def check_is_array_or_df(x):
    return isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)