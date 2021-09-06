from typing import Union, Callable, Optional
from libfairness.fairness_problem import FairnessProblem
from .checks import *
from ..utils.dataclasses import IndicesInput


def from_pandas(
    dataframe: pd.DataFrame,
    target: Union[str, pd.DataFrame, pd.Series, None],
    model: Optional[Callable] = None,
) -> IndicesInput:
    cols = set(dataframe.columns)
    if target is None:
        assert model is not None, "model must be defined when target is None"
        x = dataframe
        y = None
    elif isinstance(target, str):
        x = dataframe[cols - {target}]
        y = dataframe[target]
    elif isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
        x = dataframe
        y = pd.DataFrame(target)
    else:
        raise RuntimeError("type of target must be Dataframe, Series, str or None")
    return IndicesInput(x=x, y=y, model=model)


def from_numpy(x, y, feature_names=None):
    df = pd.DataFrame(x, columns=feature_names)
    # build dataframe
    target = pd.DataFrame(y, columns=["target"])
    return from_pandas(df, target, feature_names)


def create_fairness_problem(
    inputs=None,
    columns=None,
    function=None,
    outputs=None,
    labels=None,
    groups_studied=[],
    categorical_features=[],
):
    """Create a fairness problem object.

    Args:
        inputs ([type], optional): Inputs of the fairness problem. Defaults to None.
        columns ([type], optional): Names of the columns of the inputs. Defaults to None.
        function ([type], optional): Function of the fairness problem. Defaults to None.
        outputs ([type], optional): Outputs of the fairness problem. Defaults to None.
        labels ([type], optional): Labels of the fairness problem. Defaults to None.
        groups_studied (list, optional): List to inform which variables should be studied. Defaults to [].
        categorical_features (list, optional): List to inform which variables should be binarize. Defaults to [].

    Returns:
        FairnessProblem: An object representing the problem inwhich are saved usefull datas.
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

    return FairnessProblem(
        inputs, columns, function, outputs, labels, groups_studied, categorical_features
    )
