import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from fairness_problem import FairnessProblem

sns.set_theme(style="white", context="notebook")
rs = np.random.RandomState(8)
custom_palette = sns.diverging_palette(150, 10, s=80, l=55, n=2)
def visu_disparate_impact(fairness_problem):
    sns.set_theme(style="white", context="notebook")

# Set up the matplotlib figure
f, (ax1) = plt.subplots(1, 1, sharex=True)

# Generate some sequential data
x = np.array(list("ABCDEFGHIJ"))
y1 = np.arange(1, 11)

# Center the data to make it diverging
y2 = y1 - 5.5
sns.barplot(x=x, y=y2, palette=custom_palette, ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Sensitive variable")


# Randomly reorder the data to make it qualitative
y3 = rs.choice(y1, len(y1), replace=False)

sns.despine(bottom=True)
#plt.setp(f.axes, yticks=[])
#plt.tight_layout(h_pad=2)

plt.show()
