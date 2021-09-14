import unittest
import pandas as pd
import sklearn.datasets as ds
from libfairness.data_management.factory import from_pandas
from libfairness.data_management.processing import one_hot_encode

# from libfairness.utils.targets import y_true


class MyTestCase(unittest.TestCase):
    def test_onehot_encoding(self):
        data = ds.load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.DataFrame(data.target, columns=["target"])
        indices_inputs = from_pandas(
            df,
            target,
            None,
        )
        indices_inputs = one_hot_encode(indices_inputs, ["CHAS", "RAD"])
        self.assertEqual(
            len(df.columns),
            len(indices_inputs.variable_groups),
            "checking number of columns failed",
        )
        self.assertEqual(
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


if __name__ == "__main__":
    unittest.main()
