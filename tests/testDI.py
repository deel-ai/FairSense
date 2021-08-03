import unittest

import libfairness.data_management.factory
import libfairness.indices.standard_metrics
import numpy as np


class TestSobol(unittest.TestCase):

    def test_disparate_impact(self):

        # data
        inputs = np.array([[1., 0.], [1., 0.], [1., 1.], [1., 1.], [0., 1.], [1., 0.], [1., 1.], [0., 1.], [1., 1.], [
            1., 1.], [0., 0.], [0., 0.], [0., 1.], [1., 0.], [1., 1.], [1., 1.], [0., 1.], [1., 1.], [0., 0.], [1., 1.]])
        outputs = np.array([[0], [1], [0], [1], [1], [0], [1], [1], [0], [0], [
                           0], [1], [0], [0], [1], [1], [1], [0], [1], [0]])

        columns = ["Male", "Driver_licence"]

        gs1 = [["Male"], ["Driver_licence"]]
        gs2 = [[0], [1]]
        gs3 = [[1]]

        # results that we must obtain
        result_hard = {'Male': 1.857142857142857,
                       'Driver_licence': 0.7959183673469388}
        result_hard_3 = {'Driver_licence': 0.7959183673469388}
        # fairness problem
        my_problem = libfairness.data_management.factory.create_fairness_problem(
            inputs=inputs, outputs=outputs, groups_studied=gs1, columns=columns)
        libfairness.indices.standard_metrics.compute_disparate_impact(
            my_problem)
        self.assertEqual(my_problem.get_result(), result_hard)

        my_problem.set_groups_studied(gs2)
        libfairness.indices.standard_metrics.compute_disparate_impact(
            my_problem)
        self.assertEqual(my_problem.get_result(), result_hard)

        my_problem.set_groups_studied(gs3)
        libfairness.indices.standard_metrics.compute_disparate_impact(
            my_problem)
        self.assertEqual(my_problem.get_result(), result_hard_3)


if __name__ == '__main__':
    unittest.main()
