import sklearn.datasets as ds
import matplotlib.pyplot as plt
from libfairness.data_management.factory import from_numpy
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

if __name__ == "__main__":
    # load data
    data = ds.load_boston()
    model = DecisionTreeRegressor()
    # construct IndicesInput object
    indices_inputs = from_numpy(
        x=data.data,
        y=data.target,
        feature_names=data.feature_names,
        model=model.predict,
        target=y_true,
    )
    # apply one hot encoding
    indices_inputs = one_hot_encode(indices_inputs, ["CHAS", "RAD"])
    # build and train a model
    model.fit(indices_inputs.x, indices_inputs.y_true)
    indices_inputs_2 = IndicesInput(
        model=model.predict,
        x=indices_inputs.x,
        y_true=indices_inputs.y_true,
        objective=y_true,
    )
    di_with_ci = with_confidence_intervals(n_splits=10)(disparate_impact)
    cvm_with_ci = with_confidence_intervals(n_splits=10)(cvm_indices)
    sobol_with_ci = with_confidence_intervals(n_splits=10)(sobol_indices)
    # compute indices
    indices_outputs = (
        di_with_ci(indices_inputs_2)
        + cvm_with_ci(indices_inputs_2)
        + sobol_with_ci(indices_inputs_2, n=5000)
    )
    # display_result
    print(indices_outputs.values)
    print(format_with_intervals(indices_outputs, quantile=0.1))
    # plot results
    ax = cat_plot(indices_outputs, plot_per="index", kind="bar")
    plt.show()
