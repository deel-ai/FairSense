from libfairness.fairness_problem import FairnessProblem
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def binarize(fairness_problem:FairnessProblem, categorical_features=None):
    """Binarize the columns that have been indicated as categorical.

    Args:
        fairness_problem (FairnessProblem): The fairness problem created that contains inputs to binarize.
        categorical_features (List, optional): List containing columns to binarize. Defaults to None.

    Raises:
        ValueError: No categorical_features list set.
    """
    if categorical_features != None:
        fairness_problem.set_categorical_features(categorical_features)
    cf = fairness_problem.get_categorical_features()
    if cf == None:
        raise ValueError("FairnessProblem.categorical_features is not set yet.")
    for elt in cf:
        c = fairness_problem.get_inputs()[:,elt]
        c = np.reshape(c, (len(c),1))
        result, new_names = __one_hot_enc(c, fairness_problem.get_columns()[elt])
        fairness_problem.set_inputs(np.concatenate((fairness_problem.get_inputs(), result), axis=1))
        fairness_problem.get_columns().extend(new_names)
    index_shift = 0
    new_inputs = fairness_problem.get_inputs()
    for elt in cf:
        del fairness_problem.get_columns()[elt-index_shift]
        new_inputs = np.delete(new_inputs, elt-index_shift, 1)
        index_shift += 1
    fairness_problem.set_inputs(new_inputs)
    fairness_problem.set_categorical_features([])

def __one_hot_enc(column:np.ndarray, name):
    """Binarize a column

    Args:
        column (np.ndarray): A column of inputs
        name ([type]): The name linked to the column

    Returns:
        np.ndarray, str: The result of binarization.
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(column)
    res = encoder.transform(column).toarray()
    cat = encoder.get_feature_names([name])
    return res, cat