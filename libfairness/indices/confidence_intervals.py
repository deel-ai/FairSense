import pandas as pd
from libfairness.utils.dataclasses import IndicesInput, IndicesOutput
from sklearn.model_selection import KFold


def with_confidence_intervals(n_splits=31, shuffle=False, random_state=None):

    kf = KFold(n_splits, shuffle=shuffle, random_state=random_state)

    def confidence_computation_fct(function):
        def call_function(inputs: IndicesInput, *args, **kwargs):
            # get full inputs
            x = inputs.x
            y = inputs.y
            fold_results = []
            # repeat indices computation on each fold
            for split1, _ in kf.split(x, y):
                # build input for the fold
                x_fold = x.iloc[split1]
                y_fold = y.iloc[split1]
                fold_inputs = IndicesInput(
                    model=inputs.model,
                    x=x_fold,
                    y=y_fold,
                    variable_groups=inputs.variable_groups,
                )
                # compute the result for the fold
                fold_results.append(function(fold_inputs, *args, **kwargs))
            # merge results to compute values and confidence intervals
            fvalues = [f.values for f in fold_results]
            runs = pd.concat(fvalues)
            return IndicesOutput(runs)

        return call_function

    return confidence_computation_fct
