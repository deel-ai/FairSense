import sklearn.datasets as ds
import matplotlib.pyplot as plt
from libfairness.data_management.factory import from_numpy
from libfairness.data_management.processing import one_hot_encode
from libfairness.indices.confidence_intervals import with_confidence_intervals
from libfairness.indices.cvm import cvm_indices
from libfairness.indices.standard_metrics import disparate_impact
from libfairness.utils.dataclasses import IndicesInput
from libfairness.utils.targets import y_true as datatarget, squared_error, predictions
from libfairness.visualization.plots import cat_plot
from libfairness.visualization.text import format_with_intervals
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    # load data
    data = ds.load_boston()
    # construct IndicesInput object
    indices_inputs = from_numpy(data.data, data.target, data.feature_names,
                                target=squared_error)
    # apply one hot encoding
    indices_inputs = one_hot_encode(indices_inputs, ["CHAS", "RAD"])
    # build and train a model
    model = DecisionTreeRegressor()
    model.fit(indices_inputs.x, indices_inputs.y)
    indices_inputs_2 = IndicesInput(
        model=model.predict, x=indices_inputs.x, y=indices_inputs.y,
        target=squared_error
    )
    di_with_ci = with_confidence_intervals(n_splits=31)(disparate_impact)
    cvm_with_ci = with_confidence_intervals(n_splits=31)(cvm_indices)
    # compute indices
    indices_outputs = di_with_ci(indices_inputs_2) + cvm_with_ci(indices_inputs_2)
    # display_result
    print(format_with_intervals(indices_outputs))
    # plot results
    ax = cat_plot(indices_outputs, plot_per="index", kind="box")
    plt.show()
