import pandas as pd
import numpy as np
import sklearn.datasets as ds
from deel.fairsense.data_management.factory import from_pandas
from deel.fairsense.data_management.processing import one_hot_encode


def test_onehot_encoding():
    data = ds.load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.DataFrame(data.target, columns=["target"])
    indices_inputs = from_pandas(
        df,
        target,
        None,
    )
    indices_inputs = one_hot_encode(indices_inputs, ["CHAS", "RAD"])
    np.testing.assert_equal(
        len(df.columns),
        len(indices_inputs.variable_groups),
        "checking number of columns failed",
    )
    np.testing.assert_equal(
        indices_inputs.variable_groups,
        [
            ["CRIM"],
            ["ZN"],
            ["INDUS"],
            ["CHAS=1.0"],
            ["NOX"],
            ["RM"],
            ["AGE"],
            ["DIS"],
            [
                "RAD=2.0",
                "RAD=3.0",
                "RAD=4.0",
                "RAD=5.0",
                "RAD=6.0",
                "RAD=7.0",
                "RAD=8.0",
                "RAD=24.0",
            ],
            ["TAX"],
            ["PTRATIO"],
            ["B"],
            ["LSTAT"],
        ],
        "mismach in the output groups",
    )
