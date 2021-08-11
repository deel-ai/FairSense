from libfairness.data_management import utils
from libfairness.fairness_problem import FairnessProblem

import numpy as np

# ---- DISPARATE IMPACT ----


def __disparate_impact(fairness_problem: FairnessProblem, S_index: int):
    """Compute the disparate impact for one sensitive variable.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
        S_index (int): Index of the column where the sensitive variable
        can be found in FairnessProblem.inputs.

    Returns:
        float: Return the disparate impact of a sensitive variable.
        Disparate impact is defined as P(f(X)=1|S=0) / P(f(X)=1|S=1).
        When P(f(X)=1|S=1)=0, the function returns None.
    """
    inputs = fairness_problem.get_inputs()
    outputs = fairness_problem.get_outputs()
    a = len(np.where(inputs[:,S_index]==0)[0])
    b = len(np.where(inputs[:,S_index]==1)[0])
    if a==0 or b==0: return None
    numerator = len(np.where((outputs[:,0]==1) & (inputs[:,S_index]==0))[0]) / a
    denominator = len(np.where((outputs[:,0]==1) & (inputs[:,S_index]==1))[0]) / b
    if denominator == 0: return None
    return numerator/denominator


def __check_arg_disparate_impact(fairness_problem: FairnessProblem):
    """Check if the arguments needed to compute the disparate impact are correct.
    Rectify when it is possible, raise an exception otherwise.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
    """
    if fairness_problem.get_outputs() is None:
        utils.compute_outputs(fairness_problem)


def compute_disparate_impact(fairness_problem: FairnessProblem):
    """Compute the disparate impact for all sensitive variables S in FairnessProblem.groups_studied.
    Set FairnessProblem.result as a dict inwhich keys are the sensitive variables and
    the values are the disparate inpact values.
    Disparate impact is defined as P(f(X)=1|S=0) / P(f(X)=1|S=1).
    When P(f(X)=1|S=1)=0 for a sensitive variable, the associate result will be None.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
    """
    __check_arg_disparate_impact(fairness_problem)
    result = {}
    for var in fairness_problem.get_groups_studied():
        if type(var[0])==str:
            try:
                index_sensitive_var = fairness_problem.get_columns().index(var[0])
            except ValueError:
                raise ValueError("Verify that names you gave to FairnessProblem.groups_studied are correct : " + str(var[0]))
        else:
            index_sensitive_var = int(var[0])
        result[fairness_problem.get_columns()[index_sensitive_var]] = __disparate_impact(fairness_problem, index_sensitive_var)
    fairness_problem.set_result(result)

