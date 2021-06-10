from fairness_problem import FairnessProblem


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