import libfairness.indices.standard_metrics
import numpy as np
import libfairness.data_management.factory
from libfairness.visualization.standard_metrics_visu import visu_disparate_impact

# data
inputs = np.array(
    [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ]
)
outputs = np.array(
    [
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1],
        [0],
        [1],
        [0],
    ]
)

columns = ["Male", "Driver_licence"]

gs1 = [["Male"], ["Driver_licence"]]
gs2 = [[0], [1]]

# fairness problem
my_problem = libfairness.data_management.factory.create_fairness_problem(
    inputs=inputs, outputs=outputs, groups_studied=gs1, columns=columns
)
libfairness.indices.standard_metrics.compute_disparate_impact(my_problem)
visu_disparate_impact(my_problem)


# How to understand the result :
# Non-male are advantaged
# People with DL are advantaged
