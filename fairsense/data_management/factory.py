"""
This module contains factory functions that allow to build more easily IndicesInput
objects.
"""
from typing import Union, Callable, Optional
import pandas as pd
from fairsense.utils.dataclasses import IndicesInput


def from_pandas(
    dataframe: pd.DataFrame,
    y: Union[str, pd.DataFrame, pd.Series, None],
    model: Optional[Callable] = None,
    target: Callable = None,
) -> IndicesInput:
    """
    Builds IndicesInput from pandas dataframe.

    Args:
        dataframe: DataFrame containing the samples to analyse.
        y: Union[str, pd.DataFrame, pd.Series, None] : when str, refers to the name
            of the columns containing the labels. Must be present in dataframe. When
            pd.DataFrame or pd.Series the label are provided in the same order as in
            dataframe. When None, no labels are provided.
        model: function that can be applied on dataframe, that return a series with
            same shape as y.
        target: one of the target from the utils.fairness_objective module.

    Returns:
        an IndicesInput object that can be used to compute sensitivity indices.

    """
    cols = set(dataframe.columns)
    if y is None:
        assert model is not None, "model must be defined when target is None"
        x = dataframe
        y = None
    elif isinstance(y, str):
        x = dataframe[cols - {y}]
        y = dataframe[y]
    elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        x = dataframe
        y = pd.DataFrame(y)
    else:
        raise RuntimeError("type of target must be Dataframe, Series, str or None")
    return IndicesInput(x=x, y_true=y, model=model, objective=target)


def from_numpy(x, y, feature_names=None, model=None, target=None) -> IndicesInput:
    """
    Builds IndicesInput from numpy array.

    Args:
        x: numpy array containing the samples to analyse.
        y: numpy array containing the labels. Can be None if no labels are provided.
        feature_names: a list of str containing the features name of x. When None
            features are named with numbers.
        model: function that can be applied on dataframe, that return an series with
            same shape as y.
        target: one of the target from the utils.fairness_objective module.

    Returns:
        an IndicesInput object that can be used to compute sensitivity indices.

    """
    df = pd.DataFrame(x, columns=feature_names)
    # build dataframe
    y = pd.DataFrame(y, columns=["target"])
    return from_pandas(dataframe=df, y=y, model=model, target=target)
