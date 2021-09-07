import sklearn.datasets as ds
import matplotlib.pyplot as plt
from libfairness.data_management.factory import from_numpy
from libfairness.data_management.processing import one_hot_encode
from libfairness.indices.confidence_intervals import with_confidence_intervals
from libfairness.indices.cvm import cvm_indices
from libfairness.indices.standard_metrics import disparate_impact
from libfairness.visualization.plots import bar_plot

if __name__ == "__main__":
    # load data
    data = ds.load_boston()
    # construct IndicesInput object
    indices_inputs = from_numpy(data.data, data.target, data.feature_names)
    # apply one hot encoding
    indices_inputs = one_hot_encode(indices_inputs, ["CHAS", "RAD"])
    # compute indices
    indices_outputs = with_confidence_intervals(n_splits=31)(disparate_impact)(
        indices_inputs
    ) + with_confidence_intervals(n_splits=31)(cvm_indices)(indices_inputs)
    ax = bar_plot(indices_outputs, plot_per="index", kind="box")
    plt.show()
