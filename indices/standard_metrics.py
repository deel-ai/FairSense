from data_management import utils
from fairness_problem import FairnessProblem

# ---- DISPARATE IMPACT ----


def __disparate_impact(fairness_problem: FairnessProblem, S_index: int):
    """Compute the disparate impact for one sensitive variable.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
        S_index (int): Index of the column where the sensitive variable
        can be found in FairnessProblem.inputs.

    Returns:
        dict: Return a dict in which the keys are the values ​​of the sensitive
        variable and the values are the associated disparate impact.
        Disparate impact is defined as 𝑃(𝑓(𝑋)=1|𝑆)/𝑃(𝑓(𝑋)=1).
        When P(f(X)=1) = 0, the function returns {"all": -1.}.
    """
    occurrences = {}
    total = 0
    positive_prediction = 0
    for i in range(fairness_problem.get_outputs().shape[0]):
        temp = fairness_problem.get_inputs()[i][S_index]
        if not temp in occurrences:
            occurrences[temp] = [0, 0]
        total += 1
        output = fairness_problem.get_outputs()[i][0]
        positive_prediction += output
        (occurrences[temp])[int(output)] += 1
    result = {}
    b = positive_prediction/total
    if b == 0:
        return {"all": -1.}
    for element in occurrences:
        a = (occurrences[element])[1] / \
            ((occurrences[element])[1]+(occurrences[element])[0])
        result[element] = a/b
    return result


def __check_arg_disparate_impact(fairness_problem: FairnessProblem):
    """Check if the arguments needed to compute the disparate impact are correct.
    Rectify when it is possible, raise an exception otherwise.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
    """
    if fairness_problem.get_outputs() is None:
        utils.compute_outputs(fairness_problem)
    # TODO suite
    # verif la liste d'interet


def compute_disparate_impact(fairness_problem: FairnessProblem):
    """Compute the disparate impact for all sensitive variables S in FairnessProblem.groups_studied.
    Set FairnessProblem.result as a dict inwhich keys are the sensitive variables and
    the values are dictionaries containing the disparate impact linked to all values that S takes.
    Disparate impact is defined as 𝑃(𝑓(𝑋)=1|𝑆)/𝑃(𝑓(𝑋)=1).
    When P(f(X)=1) = 0 for a S, the associate result will be {S: {"all": -1.}}.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
    """
    __check_arg_disparate_impact(fairness_problem)
    result = {}
    for var in fairness_problem.get_groups_studied():
        result[var[0]] = __disparate_impact(fairness_problem, var[0])
    fairness_problem.set_result(result)

# ---- EQUALITY OF ODDS ----


def compute_equality_of_odds(fairness_problem: FairnessProblem):
    """Compute the equality of odds for all sensitive variables S in FairnessProblem.groups_studied.
    Set FairnessProblem.result as a dict inwhich keys are the sensitive variables and
    the values are dictionaries containing the equality of odds linked to all values that S takes.
    Equality of odds is defined as 𝑃(𝑓(𝑋)=1|𝑌=𝑖,𝑆)/𝑃(𝑓(𝑋)=1|𝑌=𝑖),𝑖=0,1.
    The result has the following shape : {S: {S_value1: (a,b)}} where
    a = 𝑃(𝑓(𝑋)=1|𝑌=0,S_value1) / 𝑃(𝑓(𝑋)=1|𝑌=0)
    and b = 𝑃(𝑓(𝑋)=1|𝑌=1,S_value1) / 𝑃(𝑓(𝑋)=1|𝑌=1).
    When an indice can't be computed, it is remplaced by -1.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
    """
    __check_arg_equality_of_odds(fairness_problem)
    result = {}
    for var in fairness_problem.get_groups_studied():
        result[var[0]] = __equality_of_odds(fairness_problem, var[0])
    fairness_problem.set_result(result)


def __check_arg_equality_of_odds(fairness_problem: FairnessProblem):
    """Check if the arguments needed to compute the equality of odds are correct.
    Rectify when it is possible, raise an exception otherwise.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
    """
    if fairness_problem.get_outputs() is None:
        utils.compute_outputs(fairness_problem)
    # TODO suite
    # verif la liste d'interet


def __equality_of_odds(fairness_problem: FairnessProblem, S_index: int):
    """Compute the equality_of_odds for one sensitive variable.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
        S_index (int): Index of the column where the sensitive variable
        can be found in FairnessProblem.inputs.

    Returns:
        dict: Return a dict in which the keys are the values ​​of the sensitive
        variable and the values are the associated equality_of_odds' tuples.
        Equality of odds is defined as 𝑃(𝑓(𝑋)=1|𝑌=𝑖,𝑆)/𝑃(𝑓(𝑋)=1|𝑌=𝑖),𝑖=0,1.
        The result has the following shape : {S_value1: (a,b)} where
        a = 𝑃(𝑓(𝑋)=1|𝑌=0,S_value1) / 𝑃(𝑓(𝑋)=1|𝑌=0)
        and b = 𝑃(𝑓(𝑋)=1|𝑌=1,S_value1) / 𝑃(𝑓(𝑋)=1|𝑌=1).
        When an indice can't be computed, it is remplaced by -1.
    """
    occurrences = {}
    total_Y0 = 0   # total number of labels equal to 0
    total_Y1 = 0   # total number of labels equal to 1
    positive_prediction_Y0 = 0   # total number of predict=1 when label=0
    positive_prediction_Y1 = 0   # total number of predict=1 when label=1
    for i in range(fairness_problem.get_outputs().shape[0]):
        # value of the sensitive variable
        tempS = fairness_problem.get_inputs()[i][S_index]
        tempL = fairness_problem.get_labels()[i][0]   # value of the label
        # value of the prediction
        tempfX = fairness_problem.get_outputs()[i][0]
        if not tempS in occurrences:
            # The dict occurrences counts the occurrences as follow :
            # [[label=0 & predict=0,label=0 & predict=1],[label=1 & predict=0,label=1 & predict=1]]
            occurrences[tempS] = [[0, 0], [0, 0]]
        if tempL == 0:
            total_Y0 += 1
            positive_prediction_Y0 += tempfX
        elif tempL == 1:
            total_Y1 += 1
            positive_prediction_Y1 += tempfX
        occurrences[tempS][int(tempL)][int(tempfX)] += 1
    result = {}
    #  Calcul of 𝑃(𝑓(𝑋)=1|𝑌=0)
    b0 = 0 if total_Y0 == 0 else positive_prediction_Y0/total_Y0
    # Calcul of 𝑃(𝑓(𝑋)=1|𝑌=1)
    b1 = 0 if total_Y1 == 0 else positive_prediction_Y1/total_Y1
    for element in occurrences:
        denum0 = occurrences[element][0][1]+occurrences[element][0][0]
        denum1 = occurrences[element][1][1]+occurrences[element][1][0]
        # Calcul of 𝑃(𝑓(𝑋)=1|𝑌=0,S)
        a0 = occurrences[element][0][1]/denum0 if denum0 != 0 else -b0
        # Calcul of 𝑃(𝑓(𝑋)=1|𝑌=1,S)
        a1 = occurrences[element][1][1]/denum1 if denum1 != 0 else -b1
        indice0 = -1 if b0 == 0 else a0/b0
        indice1 = -1 if b1 == 0 else a1/b1
        result[element] = (indice0, indice1)

    return result
