import libfairness.indices.standard_metrics
import libfairness.visualization.standard_metrics_visu
import numpy as np
import libfairness.data_management.factory


# data
inputs = np.array([[1., 0.], [1., 0.], [1., 1.], [1., 1.], [0., 1.], [1., 0.], [1., 1.], [0., 1.], [1., 1.], [
                  1., 1.], [0., 0.], [0., 0.], [0., 1.], [1., 0.], [1., 1.], [1., 1.], [0., 1.], [1., 1.], [0., 0.], [1., 1.]])
outputs = np.array([[0],[1],[0],[1],[1],[0],[1],[1],[0],[0],[0],[1],[0],[0],[1],[1],[1],[0],[1],[0]])

columns = ["Male","Driver_licence"]

gs = [["Male"],["Driver_licence"]]
#gs = [[0],[1]]
my_problem = libfairness.data_management.factory.create_fairness_problem(inputs=inputs, outputs=outputs,groups_studied=gs,columns=columns)
libfairness.indices.standard_metrics.compute_disparate_impact(my_problem)
print(my_problem.get_result())
libfairness.visualization.standard_metrics_visu.visu_disparate_impact(my_problem)

result_one = {0: {1.0: 0.7692307692307693, 0.0: 1.4285714285714286}}

t = {0: {1.0: 0.7692307692307693, 0.0: 1.4285714285714286}, 1: {0.0: 0.8571428571428571, 1.0: 1.0769230769230769}}
