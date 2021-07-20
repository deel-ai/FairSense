import unittest

import numpy as np
from libfairness.data_management import utils
from libfairness.data_management.factory import create_fairness_problem
from libfairness.fairness_problem import FairnessProblem


class TestFormatting(unittest.TestCase):

    def test_binarize(self):
        x1 = ["Romain", "Bleu", 23]
        x2 = ["Romain", "Vert", 24]
        x3 = ["David", "Vert", 20]

        c = ["Nom", "Yeux", "Age"]

        inp = np.array([x1, x2, x3])
        my_problem = create_fairness_problem(inputs=inp, columns=c)

        print("--------BEFORE---------")
        print(my_problem.get_columns())
        print(my_problem.get_inputs())
        print("--------AFTER---------")
        utils.binarize(my_problem, ["Nom", "Age"])
        print(my_problem.get_columns())
        print(my_problem.get_inputs())
        comparison = my_problem.get_inputs() == np.array([['Bleu','0.0','1.0','0.0','1.0','0.0'],
                                                   ['Vert','0.0','1.0','0.0','0.0','1.0'],
                                                   ['Vert','1.0','0.0','1.0','0.0','0.0']])

        equal_arrays = comparison.all()
        self.assertTrue(equal_arrays)
        self.assertEqual(my_problem.get_columns(), [
                         'Yeux', 'Nom_David', 'Nom_Romain', 'Age_20', 'Age_23', 'Age_24'])

if __name__ == '__main__':
    unittest.main()
