import matplotlib.pyplot as plt
import numpy as np

import json

import seaborn as sns

return_history = None

directory_list = [ "logsq", "logsq_1", "logsq_2", "logsq_3", "logsq_4", "logsq_5", "logsq_6", "logsq_7" ]

number_of_trials = 10000

preturn_history = np.zeros((60, number_of_trials))

index = 0

for directory in directory_list:
	with open(directory+'/return_history.json', 'r') as fp:
		return_history = json.load(fp)

	local_trials = len(return_history["0"])

	if index+local_trials > number_of_trials:
		local_trials = number_of_trials

	for x in return_history:
		if int(x) < 60:
			preturn_history[int(x)][index:index+local_trials] += np.array(return_history[x])

	index+=local_trials

	if index >= number_of_trials:
		break


preturn_history = preturn_history.T


# plt.plot(range(len(mean_returns)), mean_returns, linewidth=2)

# plt.show()

ax = sns.tsplot(data=preturn_history)

plt.xlim([0, 60])
plt.ylim([-1000, 0])

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.grid(True)

plt.tight_layout()
plt.show()

# sns.show()