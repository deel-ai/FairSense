import unittest

import numpy as np
import pandas as pd

from libfairness.data_management.factory import create_fairness_problem
from libfairness.indices.cvm import compute_cvm
from libfairness.visualization.cvm_visu import visu_cvm
from data_for_tests import get_data

class TestCVM(unittest.TestCase):
    
    def test_cvm(self):
        x, l, y = get_data()
        my_problem = create_fairness_problem(inputs=x, outputs=y)
        compute_cvm(my_problem)
        print(my_problem.get_result())
        visu_cvm(my_problem)
        
        #self.assertEqual()

if __name__ == '__main__':
    unittest.main()