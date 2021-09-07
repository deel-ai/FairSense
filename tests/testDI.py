import unittest

import libfairness.data_management.factory
from libfairness.indices.standard_metrics import disparate_impact
import numpy as np
import pandas as pd

from libfairness.utils.dataclasses import IndicesInput


class TestSobol(unittest.TestCase):
    def test_new(self):
        inputs = IndicesInput(
            model=None,
            x=pd.DataFrame(
                data=np.random.randint(low=0, high=2, size=(5000, 4)),
                columns="a,b,c," "" "d".split(","),
            ),
            y=pd.DataFrame(np.random.randint(low=0, high=2, size=(5000,))),
            variable_groups=[["a", "b"], ["c"], ["d"]],
        )
        outputs = disparate_impact(inputs)
        np.testing.assert_array_less(0.95, outputs.results)
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

        # results that we must obtain
        result_hard = pd.DataFrame(
            {"Male": [0.538462], "Driver_licence": [0.795918]}
        ).transpose()
        result_hard_2 = pd.DataFrame({"Driver_licence": [0.795918]}).transpose()
        # fairness problem
        index_input = IndicesInput(x=inputs, y=outputs, variable_groups=gs1)
        result = disparate_impact(index_input)
        np.testing.assert_allclose(
            result.values.values, result_hard.values, atol=1e-4, rtol=1e-4
        )

        index_input = IndicesInput(x=inputs, y=outputs, variable_groups=gs2)
        result = disparate_impact(index_input)
        np.testing.assert_allclose(
            result.values.values, result_hard_2.values, atol=1e-4, rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
