import numpy as np
import data_management.factory
import indices.sobol
import indices.cvm
import visualization.one_way_to_display

# My data
x = np.linspace(1, 10, 10)
f = lambda x: np.random.randint(0,2)
labels = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1]

my_problem = data_management.factory.create_fairness_problem(inputs=x, function=f, labels=labels)
indices.cvm(my_problem, cols=2)
visualization.one_way_to_display(my_problem)



