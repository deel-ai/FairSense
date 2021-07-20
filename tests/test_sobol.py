import unittest

import numpy as np
import pandas as pd

from libfairness.data_management.factory import create_fairness_problem
from libfairness.indices.sobol import compute_sobol
from data_for_tests import get_data
from data_for_tests import f_for_test
from libfairness.visualization.sobol_visu import visu_sobol

class TestSobol(unittest.TestCase):
    
    def test_sobol(self):
        x, l, y = get_data()
        my_problem = create_fairness_problem(inputs=x, function=f_for_test)
        compute_sobol(my_problem)
        visu_sobol(my_problem)
        #self.assertEqual()

if __name__ == '__main__':
    unittest.main()