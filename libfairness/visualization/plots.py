import seaborn as sns
from libfairness.utils.dataclasses import IndicesOutput

"""
This module contains functions used to plot the outputs of the indices. The outputs of
the indices are epresented as :mod:`.utils.dataclasses.IndicesOutput`.
"""


def cat_plot(
    indices: IndicesOutput, plot_per="variable", kind="bar", col_wrap=None, **kwargs
):
    """
    Uses the `seaborn.catplot`_ to plot the indices.

    Args:
        indices (IndicesOutput): computed indices
        plot_per (str): can be either `variable` or `index`, when set to `variable`
            there is one graph per variable, each graph showing the values of all
            indices. Respectively setting to `index` will build one graph per index,
            each showing the values for all variable.
        kind (str): kind of visualization to produce, can be one of `strip`, `swarm`,
            `box`, `violin`, `boxen`, `point`, `bar`.
        col_wrap (Optional(int)): “Wrap” the column variable at this width, so that
            the column facets span multiple rows.
        **kwargs: extra arguments passed to `seaborn.catplot`_.

    Returns:
        a matplotlib axes object

    .. _seaborn.catplot:
        https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot

    """
    assert plot_per.lower().strip() in {"variable", "index", "indices"}
    indices_names = indices.values.columns
    variable_names = indices.values.index
    data = indices.runs.stack().reset_index()
    data.rename(
        columns={"level_0": "variable", "level_1": "index", 0: "value"}, inplace=True
    )
    if plot_per == "variable":
        col = "variable"
        x = "index"
        order = indices_names
    else:
        col = "index"
        x = "variable"
        order = variable_names
    ax = sns.catplot(
        data=data,
        x=x,
        y="value",
        col=col,
        kind=kind,
        col_wrap=col_wrap,
        order=order,
        **kwargs
    )
    ax.set(ylim=(0.0, 1.0))
    ax.set_xticklabels(rotation=45, horizontalalignment="right")
    return ax
