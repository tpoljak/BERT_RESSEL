import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set(style="darkgrid") # white, dark, whitegrid, darkgrid, ticks

fig, ax1 = plt.subplots()

hue = ["MRR"]*9 + ["R@1"]*9
mrr = [0.894, 0.902, 0.904, 0.913, 0.905, 0.910, 0.906, 0.914, 0.916]
r_at_1 = [0.831, 0.845,0.844, 0.860, 0.849, 0.855, 0.849, 0.860, 0.863]
avg_len = [45.555, 58.153, 71.920, 86.520, 101.621, 117.592, 132.646, 152.7, 170.180]

performance = mrr + r_at_1
turn = ['2','3','4','5','6','7','8','9','10~'] * 2
hue_len = ["avg_len"] * 9
turn_len = ['2','3','4','5','6','7','8','9','10~']
# sns.set_ylim(80,95)
avg_ubuntu_len = pd.DataFrame(data=zip(hue_len, turn_len, avg_len), columns=["Avg lengths", "Number of utterances in dialog", "Average of lengths"])
box = sns.barplot(x="Number of utterances in dialog", y="Average of lengths", hue='Avg lengths', data=avg_ubuntu_len, palette='rocket')
ax1.legend().set_visible(False)

# ubuntu = pd.DataFrame(data=zip(hue, turn, performance), columns=["Metric", "Number of utterances in dialog", "Performance"])
# line = sns.lineplot(x="Number of utterances in dialog", y="Performance", hue='Metric', data=ubuntu, legend='brief', sort=False, ax=ax2)
# line.axes.set_ylim(0.82, 0.935)

# line.set(ylim=((80,95)))
# axes.set_ylim(80,)
# plt.legend(loc='upper left')
plt.show()
# plt.savefig('quantity_analysis.pdf')