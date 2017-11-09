import matplotlib.pyplot as plt
import numpy as np

import json

import seaborn as sns

return_history = None

returns_array = np.array([])

with open('returns.txt', 'r') as fp:
	file_out = fp.read()

file_out = file_out.split('\n')

rows = len(file_out)

# print len(file_out)

start = True;

for x in file_out:
	a = x.split(';')
	a = a[:-1]
	a = map(float, a)
	
	if start:
		returns_array = np.append(returns_array, np.array(a))
		cols = len(a)
		start = False;
	else:
		returns_array = np.append(returns_array, np.array(a), axis=0)

returns_array = returns_array.reshape(rows-1, cols)

# print returns_array
# plt.plot(range(len(mean_returns)), mean_returns, linewidth=2)

# plt.show()

ax = sns.tsplot(data=returns_array)

# plt.ylim([-1000, 0])

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.grid(True)

plt.tight_layout()
plt.show()

# sns.show()