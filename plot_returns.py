import matplotlib.pyplot as plt
import numpy as np

import json

import seaborn as sns

return_history = None

q = 3
mode = "q"

if q == 3:
	if mode == "s":
		directory = "logs"
	else:
		directory = "logsq"
elif q==2:
	if mode == "s":
		directory = "htunings"
	else:
		directory = "htuningq"

with open(directory+'/return_history.json', 'r') as fp:
	return_history = json.load(fp)

with open(directory+'/system_stats.json', 'r') as fp:
	system_stats = json.load(fp)

# Preprocess

mean_returns = np.zeros(200)

# system_stats['number_of_trials'] = 2

preturn_history = np.zeros((200, system_stats['number_of_trials']))

for x in return_history:
	mean_returns[int(x)] = np.mean(return_history[x][:system_stats['number_of_trials']])
	preturn_history[int(x)] += np.array(return_history[x][:system_stats['number_of_trials']])


preturn_history = preturn_history.T


# plt.plot(range(len(mean_returns)), mean_returns, linewidth=2)

# plt.show()

ax = sns.tsplot(data=preturn_history)

plt.ylim([-1000, 0])

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.grid(True)

plt.tight_layout()
plt.show()

# sns.show()