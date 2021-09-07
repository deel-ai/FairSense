import seaborn as sns
from libfairness.utils.dataclasses import IndicesOutput


def cat_plot(
    indices: IndicesOutput, plot_per="variable", kind="bar", col_wrap=None, **kwargs
):
    assert plot_per.lower().strip() in {"variable", "index", "indices"}
    if plot_per == "variable":
        col = "variable"
        x = "index"
    else:
        col = "index"
        x = "variable"
    data = indices.runs.stack().reset_index()
    data.rename(
        columns={"level_0": "variable", "level_1": "index", 0: "value"}, inplace=True
    )
    ax = sns.catplot(
        data=data, x=x, y="value", col=col, kind=kind, col_wrap=col_wrap, **kwargs
    )
    ax.set(ylim=(0.0, 1.0))
    return ax
