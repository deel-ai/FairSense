import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from libfairness.fairness_problem import FairnessProblem

def visu_disparate_impact(fairness_problem):
    sns.set_theme(style="white", context="notebook")
    custom_palette = {}
    res = fairness_problem.get_result().copy()
    for elt in res:
        res[elt] -= 1
        if res[elt] >= 0 :
            custom_palette[elt] = 'r'
        else : 
            custom_palette[elt] = 'g'
    sns.barplot(x=list(res.keys()), y=list(res.values()), palette=custom_palette)
    plt.axhline(0, color="k", clip_on=False)
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)
    plt.show()


def old_visu_disparate_impact(fairness_problem):
    sns.set_theme(style="white", context="notebook")
    custom_palette = sns.diverging_palette(150, 10, s=80, l=55, n=2)
    gs = fairness_problem.get_groups_studied()
    n = len(gs)
    f, axs = plt.subplots(n)
    res = fairness_problem.get_result()
    if n == 1:
        res_i = res[gs[0][0]]
        y = list(res_i.values())
        for j in range(len(y)):
            y[j] -= 1
        sns.barplot(x=list(res_i.keys()), y=y, palette=custom_palette)
        axs.axhline(0, color="k", clip_on=False)
        axs.set_ylabel(gs[0][0])
    else:
        for i in range(n):
            res_i = res[gs[i][0]]
            y = list(res_i.values())
            for j in range(len(y)):
                y[j] -= 1
            sns.barplot(x=list(res_i.keys()), y=y, palette=custom_palette, ax=axs[i])
            axs[i].axhline(0, color="k", clip_on=False)
            axs[i].set_ylabel(gs[i][0])
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)
    plt.show()
