import pandas as pd
from libfairness.utils.dataclasses import IndicesInput, IndicesOutput
from sklearn.model_selection import KFold
from tqdm import tqdm


def with_confidence_intervals(n_splits=31, shuffle=False, random_state=None):

    kf = KFold(n_splits, shuffle=shuffle, random_state=random_state)

    def confidence_computation_fct(function):
        def call_function(inputs: IndicesInput, *args, **kwargs):
            # get full inputs
            x = inputs.x
            y = inputs.y_true
            fold_results = []
            # repeat indices computation on each fold
            for _, split in tqdm(kf.split(x, y), total=n_splits, ncols=80):
                # build input for the fold
                x_fold = x.iloc[split]
                y_fold = y.iloc[split]
                fold_inputs = IndicesInput(
                    model=inputs.model,
                    x=x_fold,
                    y_true=y_fold,
                    variable_groups=inputs.variable_groups,
                    objective=inputs.objective,
                )
                # compute the result for the fold
                fold_results.append(function(fold_inputs, *args, **kwargs))
            # merge results to compute values and confidence intervals
            fvalues = [f.values for f in fold_results]
            runs = pd.concat(fvalues)
            return IndicesOutput(runs)

        return call_function

    return confidence_computation_fct
