from typing import Union, Callable, Optional
import pandas as pd
from libfairness.utils.dataclasses import IndicesInput


def from_pandas(
    dataframe: pd.DataFrame,
    y: Union[str, pd.DataFrame, pd.Series, None],
    model: Optional[Callable] = None,
    target: Callable = None,
) -> IndicesInput:
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


def from_numpy(x, y, feature_names=None, model=None, target=None):
    df = pd.DataFrame(x, columns=feature_names)
    # build dataframe
    y = pd.DataFrame(y, columns=["target"])
    return from_pandas(dataframe=df, y=y, model=model, target=target)
