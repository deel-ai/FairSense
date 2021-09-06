import unittest

import numpy as np
import pandas as pd
from libfairness.data_management import utils
from libfairness.data_management.factory import create_fairness_problem
from libfairness.fairness_problem import FairnessProblem


class TestFormatting(unittest.TestCase):
    def test_inputs_numpy_panda(self):

        x1 = [4, 5, 23]
        x2 = [76, 146, 24]
        x3 = [765, 9, 20]
        c = ["Nom", "Yeux", "Age"]

        # numpy
        inp_numpy = np.array([x1, x2, x3])
        my_problem_numpy = create_fairness_problem(inputs=inp_numpy, columns=c)

        # pandas
        inp_pandas = pd.DataFrame([x1, x2, x3], columns=c)
        my_problem_pandas = create_fairness_problem(inputs=inp_pandas)

        self.assertEqual(
            my_problem_numpy.get_columns(), my_problem_pandas.get_columns()
        )
        self.assertTrue(
            np.array_equal(
                my_problem_numpy.get_inputs(), my_problem_pandas.get_inputs()
            )
        )

    def test_binarize(self):
        x1 = ["Romain", "Bleu", 23]
        x2 = ["Romain", "Vert", 24]
        x3 = ["David", "Vert", 20]

        c = ["Nom", "Yeux", "Age"]

        inp = np.array([x1, x2, x3])
        my_problem = create_fairness_problem(inputs=inp, columns=c)

        utils.binarize(my_problem, ["Nom", "Age"])
        comparison = my_problem.get_inputs() == np.array(
            [
                ["Bleu", "0.0", "1.0", "0.0", "1.0", "0.0"],
                ["Vert", "0.0", "1.0", "0.0", "0.0", "1.0"],
                ["Vert", "1.0", "0.0", "1.0", "0.0", "0.0"],
            ]
        )

        equal_arrays = comparison.all()
        self.assertTrue(equal_arrays)
        self.assertEqual(
            my_problem.get_columns(),
            ["Yeux", "Nom_David", "Nom_Romain", "Age_20", "Age_23", "Age_24"],
        )


if __name__ == "__main__":
    unittest.main()
