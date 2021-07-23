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
        res_df = pd.DataFrame([["0.06766917293233082",0.0],[0.20300751879699247,0.0],[0.3082706766917293,0.0],[0.18796992481203006,0.0]],columns=["CVM","CVM_indep"])
        
        print(my_problem.get_result().eq(res_df))
        self.assertTrue(my_problem.get_result().equals(res_df))

if __name__ == '__main__':
    unittest.main()