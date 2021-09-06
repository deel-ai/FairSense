import unittest

import numpy as np
import pandas as pd
from libfairness.data_management.factory import create_fairness_problem
from libfairness.indices.sobol import compute_sobol

from tests.test_sensivity_indices import gaussian_data_generator


class TestSobol(unittest.TestCase):
    def test_sobol(self):
        # Setup
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)

        nsample = 5 * 10 ** 2
        data_sample = 10 ** 4
        bootstrap_size = 150

        # Data
        def func(x):
            return np.sum(x, axis=1)

        data = gaussian_data_generator(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, N=data_sample
        )

        # Use Case
        my_problem = create_fairness_problem(inputs=data, function=func)
        compute_sobol(my_problem, n=nsample, bs=bootstrap_size)

        result = my_problem.get_result()[["S", "ST", "S_ind", "ST_ind"]].to_numpy()
        result_hard = np.array(
            [
                [0.33, 0.34, 0.33, 0.33],
                [0.33, 0.33, 0.33, 0.33],
                [0.34, 0.34, 0.34, 0.34],
            ]
        )
        self.assertTrue(np.allclose(result, result_hard, atol=0.03))


if __name__ == "__main__":
    unittest.main()
