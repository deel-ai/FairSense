import indices.standard_metrics
import visualization.standard_metrics_visu
import numpy as np
import data_management.factory

x = np.zeros((20,2))
for i in range(x.shape[0]):
    x[i][0] = np.random.randint(0,3)
    x[i][1] = np.random.randint(0,2)
print(x)

y = np.zeros((20,1))
for i in range(y.shape[0]):
    y[i] = np.random.randint(0,2)

#gs = [[0]]
gs = [[0],[1]]
my_problem = data_management.factory.create_fairness_problem(inputs=x, outputs=y,groups_studied=gs)
indices.standard_metrics.compute_disparate_impact(my_problem)
print(my_problem.get_result())
visualization.standard_metrics_visu.visu_disparate_impact(my_problem)