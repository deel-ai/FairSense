import unittest
from libfairness.indices.standard_metrics import disparate_impact
from libfairness.utils.fairness_objective import y_true
import numpy as np
import pandas as pd

from libfairness.utils.dataclasses import IndicesInput


class TestSobol(unittest.TestCase):
    def test_new(self):
        inputs = IndicesInput(
            model=None,
            x=pd.DataFrame(
                data=np.random.randint(low=0, high=2, size=(10000, 4)),
                columns="a,b,c," "" "d".split(","),
            ),
            y_true=pd.DataFrame(np.random.randint(low=0, high=2, size=(10000,))),
            variable_groups=[["a", "b"], ["c"], ["d"]],
            objective=y_true,
        )
        outputs = disparate_impact(inputs)
        np.testing.assert_array_less(outputs.values, 0.05)
        print(outputs)

    def test_disparate_impact(self):

        columns = ["Male", "Driver_licence"]
        # data
        inputs = pd.DataFrame(
            np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            columns=columns,
        )
        outputs = pd.DataFrame(
            np.array(
                [
                    [0],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                ]
            ),
            columns=["outputs"],
        )

        gs1 = [["Male"], ["Driver_licence"]]
        gs2 = [["Driver_licence"]]
        # fairness problem
        index_input = IndicesInput(
            x=inputs, y_true=outputs, variable_groups=gs1, objective=y_true
        )
        result = disparate_impact(index_input)
        self.assertAlmostEqual(result.values.loc["Male"]["DI"], 1 - 0.538462, places=4)
        self.assertAlmostEqual(
            result.values.loc["Driver_licence"]["DI"], 1 - 0.795918, places=4
        )

        index_input = IndicesInput(
            x=inputs, y_true=outputs, variable_groups=gs2, objective=y_true
        )
        result = disparate_impact(index_input)
        self.assertAlmostEqual(
            result.values.loc["Driver_licence"]["DI"], 1 - 0.795918, places=4
        )


if __name__ == "__main__":
    unittest.main()
