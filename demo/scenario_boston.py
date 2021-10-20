import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libfairness.data_management.factory import from_numpy, from_pandas
from libfairness.data_management.processing import one_hot_encode
from libfairness.indices.confidence_intervals import with_confidence_intervals
from libfairness.indices.cvm import cvm_indices
from libfairness.indices.standard_metrics import disparate_impact
from libfairness.indices.sobol import sobol_indices
from libfairness.utils.dataclasses import IndicesInput
from libfairness.utils.fairness_objective import y_true, squared_error, y_pred
from libfairness.visualization.plots import cat_plot
from libfairness.visualization.text import format_with_intervals
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("data/adult.csv")
data["income"] = data["income"] == ">50K"
data.head()

indices_inputs = from_pandas(data, "income", target=y_true)
categorical_cols = list(filter(lambda col: data.dtypes[col] == "O", data.columns))
indices_inputs = one_hot_encode(indices_inputs, categorical_cols)

indices_outputs = disparate_impact(indices_inputs)

di_with_ci = with_confidence_intervals(n_splits=2)(disparate_impact)
indices_outputs = di_with_ci(indices_inputs)
