import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from libfairness.fairness_problem import FairnessProblem

def visu_cvm(fairness_problem : FairnessProblem):
    res = fairness_problem.get_result()[["CVM", "CVM_indep"]]
    res = res.reset_index()
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1,2, sharey="all")
    fig.suptitle("CVM")
    axs[0].set(ylim=(0.0, 1.0))
    sns.barplot(x="index",y="CVM", data=res, ax=axs[0])
    sns.barplot(x="index",y="CVM_indep", data=res, ax=axs[1])
    plt.show()