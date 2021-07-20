import numpy as np
from libfairness.data_management import utils
from libfairness.data_management.factory import create_fairness_problem
from libfairness.fairness_problem import FairnessProblem

x1 = ["Romain", "Bleu", 23]
x2 = ["Romain", "Vert", 24]
x3 = ["David", "Vert", 20]

c = ["Nom", "Yeux", "Age"]
#c = None

inp = np.array([x1,x2,x3])
my_problem = create_fairness_problem(inputs=inp, columns=c)

print("--------BEFORE---------")
print(my_problem.get_columns())
print(my_problem.get_inputs())
print("--------AFTER---------")
utils.binarize(my_problem,["Nom","Age"])
print(my_problem.get_columns())
print(my_problem.get_inputs())
