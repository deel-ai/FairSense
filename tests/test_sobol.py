import unittest
import numpy as np
import pandas as pd
from libfairness.indices.sobol import sobol_indices
from libfairness.utils.dataclasses import IndicesInput
from libfairness.utils.fairness_objective import y_pred


def gaussian_data_generator(sigma12, sigma13, sigma23, N, var1=1.0, var2=1.0, var3=1.0):
    cov = np.mat(
        [[var1, sigma12, sigma13], [sigma12, var2, sigma23], [sigma13, sigma23, var3]]
    )
    x = np.random.multivariate_normal(mean=np.array([0, 0, 0]), cov=cov, size=N)
    return pd.DataFrame(x)


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)
        self.atol = 0.05
        self.rtol = 0.5  # high rtol for low values
        self.nsample = 10 ** 4
        self.data_sample = 10 ** 3
        # test preparation, do all computations, and results are stored in two table
        # (row are x_i and columns are indices)
        # create tables from 4.1
        func = lambda x: np.sum(x, axis=1)
        data = gaussian_data_generator(
            sigma12=0.5, sigma13=0.8, sigma23=0, N=self.data_sample
        )
        self.indices_table = sobol_indices(
            IndicesInput(model=func, x=data, objective=y_pred), n=self.nsample
        )

        func = lambda x: np.sum(x, axis=1)
        data = gaussian_data_generator(
            sigma12=-0.5, sigma13=0.2, sigma23=-0.7, N=self.data_sample
        )
        self.indices_table_2 = sobol_indices(
            IndicesInput(model=func, x=data, objective=y_pred), n=self.nsample
        )

        func = lambda x: np.sum(x, axis=1)
        data = gaussian_data_generator(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, N=self.data_sample
        )
        self.indices_table_3 = sobol_indices(
            IndicesInput(model=func, x=data, objective=y_pred), n=self.nsample
        )

        func = lambda x: x[:, 0]
        data = gaussian_data_generator(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, N=self.data_sample
        )
        self.indices_table_4 = sobol_indices(
            IndicesInput(model=func, x=data, objective=y_pred), n=self.nsample
        )

    def test_sobol(self):
        # check S match the value of the paper
        sobol_1 = self.indices_table.values.values[:, 0]
        sobol_2 = self.indices_table_2.values.values[:, 0]
        sobol_3 = self.indices_table_3.values.values[:, 0]
        sobol_4 = self.indices_table_4.values.values[:, 0]
        np.testing.assert_allclose(
            sobol_1,
            [0.94, 0.40, 0.58],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 1",
        )
        np.testing.assert_allclose(
            sobol_2,
            [0.49, 0.04, 0.25],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_3,
            [0.33, 0.33, 0.33],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_4,
            [1.0, 0.0, 0.0],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )

    def test_sobol_total(self):
        # check ST match the value of the paper
        sobol_1 = self.indices_table.values.values[:, 1]
        sobol_2 = self.indices_table_2.values.values[:, 1]
        sobol_3 = self.indices_table_3.values.values[:, 0]
        sobol_4 = self.indices_table_4.values.values[:, 0]
        np.testing.assert_allclose(
            sobol_1,
            [0.94, 0.40, 0.58],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol total] did not match expected values for linear case 1",
        )
        np.testing.assert_allclose(
            sobol_2,
            [0.49, 0.04, 0.25],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol total] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_3,
            [0.33, 0.33, 0.33],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_4,
            [1.0, 0.0, 0.0],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )

    def test_sobol_ind(self):
        # check S_i match the value of the paper
        sobol_1 = self.indices_table.values.values[:, 2]
        sobol_2 = self.indices_table_2.values.values[:, 2]
        sobol_3 = self.indices_table_3.values.values[:, 0]
        sobol_4 = self.indices_table_4.values.values[:, 0]
        np.testing.assert_allclose(
            sobol_1,
            [0.02, 0.05, 0.03],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol ind] did not match expected values for linear case 1",
        )
        np.testing.assert_allclose(
            sobol_2,
            [0.72, 0.37, 0.48],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol ind] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_3,
            [0.33, 0.33, 0.33],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_4,
            [1.0, 0.0, 0.0],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )

    def test_sobol_total_ind(self):
        # check ST_i match the value of the paper
        sobol_1 = self.indices_table.values.values[:, 3]
        sobol_2 = self.indices_table_2.values.values[:, 3]
        sobol_3 = self.indices_table_3.values.values[:, 0]
        sobol_4 = self.indices_table_4.values.values[:, 0]
        np.testing.assert_allclose(
            sobol_1,
            [0.02, 0.05, 0.03],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol total ind] did not match expected values for linear case 1",
        )
        np.testing.assert_allclose(
            sobol_2,
            [0.72, 0.37, 0.48],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol total ind] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_3,
            [0.33, 0.33, 0.33],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )
        np.testing.assert_allclose(
            sobol_4,
            [1.0, 0.0, 0.0],
            atol=self.atol,
            rtol=self.rtol,
            err_msg="indicator [sobol] did not match expected values for linear case 2",
        )


if __name__ == "__main__":
    unittest.main()
