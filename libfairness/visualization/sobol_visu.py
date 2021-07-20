import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from libfairness.fairness_problem import FairnessProblem

def visu_sobol(fairness_problem : FairnessProblem):
    res = fairness_problem.get_result()[["S", "ST", "S_ind", "ST_ind"]]
    res = res.reset_index()
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2,2, sharey="all")
    fig.suptitle("Sobol")
    axs[0,0].set(ylim=(0.0, 1.0))
    sns.barplot(x="index",y="S", data=res, ax=axs[0,0])
    sns.barplot(x="index",y="ST", data=res, ax=axs[0,1])
    sns.barplot(x="index",y="S_ind", data=res, ax=axs[1,0])
    sns.barplot(x="index",y="ST_ind", data=res, ax=axs[1,1])
    plt.show()
    