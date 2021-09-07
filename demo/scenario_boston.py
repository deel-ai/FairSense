import sklearn.datasets as ds
import matplotlib.pyplot as plt
from libfairness.data_management.factory import from_numpy
from libfairness.data_management.processing import one_hot_encode
from libfairness.indices.confidence_intervals import with_confidence_intervals
from libfairness.indices.cvm import cvm_indices
from libfairness.indices.standard_metrics import disparate_impact
from libfairness.visualization.plots import cat_plot

if __name__ == "__main__":
    # load data
    data = ds.load_boston()
    # construct IndicesInput object
    indices_inputs = from_numpy(data.data, data.target, data.feature_names)
    # apply one hot encoding
    indices_inputs = one_hot_encode(indices_inputs, ["CHAS", "RAD"])
    # the wrapper with_confidence_intervals allows to compute CI for any index
    di_with_ci = with_confidence_intervals(n_splits=31)(disparate_impact)
    cvm_with_ci = with_confidence_intervals(n_splits=31)(cvm_indices)
    # compute indices
    indices_outputs = di_with_ci(indices_inputs) + cvm_with_ci(indices_inputs)
    # plot results
    ax = cat_plot(indices_outputs, plot_per="index", kind="box")
    plt.show()
