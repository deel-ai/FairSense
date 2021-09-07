from typing import Union, Callable, Optional
import pandas as pd
from libfairness.utils.dataclasses import IndicesInput


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


def from_numpy(x, y, feature_names=None, model=None):
    df = pd.DataFrame(x, columns=feature_names)
    # build dataframe
    target = pd.DataFrame(y, columns=["target"])
    return from_pandas(df, target, model)
