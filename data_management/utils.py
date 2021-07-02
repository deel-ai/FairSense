from fairness_problem import FairnessProblem
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def compute_outputs(fairness_problem: FairnessProblem):
    """Compute the outputs of the fairness problem from FairnessProblem.inputs
    and FairnessProblem.function.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.

    Raises:
        ValueError: [description]
    """
    if fairness_problem.get_function() is None or fairness_problem.get_inputs() is None:
        raise ValueError("FairnessProblem.inputs or FairnessProblem.function is not set yet.")
    # TODO calcul des outputs

def binarize(fairness_problem:FairnessProblem, categorical_features=None):
    """[summary]

    Args:
        fairness_problem (FairnessProblem): [description]
        categorical_features ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]
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
    """[summary]

    Args:
        column (np.ndarray): [description]
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(column)
    res = encoder.transform(column).toarray()
    cat = encoder.get_feature_names([name])
    return res, cat