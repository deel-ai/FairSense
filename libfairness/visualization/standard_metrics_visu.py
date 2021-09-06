import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from libfairness.fairness_problem import FairnessProblem


def visu_disparate_impact(fairness_problem):
    """Display the result attribute after computing disparate inpact.

    Args:
        fairness_problem (FairnessProblem): The fairness problem studied.
    """
    sns.set_theme(style="white", context="notebook")
    custom_palette = {}
    res = fairness_problem.get_result().copy()
    for elt in res:
        res[elt] -= 1
        if res[elt] >= 0:
            custom_palette[elt] = "r"
        else:
            custom_palette[elt] = "g"
    sns.barplot(x=list(res.keys()), y=list(res.values()), palette=custom_palette)
    plt.axhline(0, color="k", clip_on=False)
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)
    plt.show()
