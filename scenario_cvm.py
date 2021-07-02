import numpy as np
import pandas as pd

from data_management.factory import create_fairness_problem
from indices.cvm import compute_cvm
from indices.test_sensivity_indices import gaussian_data_generator
from visualization.visu_demo_cvm import visu_demo_cvm

if __name__ == '__main__':
    # Setup + Data
    samples = 10000
    cols = ["X_{}".format(i) for i in range(5)]
    x = pd.DataFrame(np.random.normal(size=(samples, 5)), columns=cols)
    y = pd.DataFrame()
    y["Y"] = x["X_0"] + 2 * x["X_1"] - x["X_2"] + \
        np.random.normal(0, 0.005, len(x))

    # Use Case
    my_problem = create_fairness_problem(inputs=x, outputs=y.to_numpy())
    compute_cvm(my_problem)
    visu_demo_cvm(my_problem)
