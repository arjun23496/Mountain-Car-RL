from __future__ import division

from environment import Environment
from agent import Agent
from agent import FourierExpansion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mode = "q"

number_of_values = 100
show_every = 5

env_obj = Environment()
agent_obj = Agent(env_obj)
fourier = FourierExpansion(1)

if mode=="q":
	directory = "logsq"
elif mode == "s":
	directory = "logs"

with open(directory+'/learned_weights.pkl', 'r') as fp:
	agent_obj.w = np.load(fp)

print agent_obj.w

x_values = np.arange(env_obj.position_lbound, env_obj.position_ubound, (env_obj.position_ubound-env_obj.position_lbound)/number_of_values)
v_values = np.arange(env_obj.velocity_lbound, env_obj.velocity_ubound, (env_obj.velocity_ubound-env_obj.velocity_lbound)/number_of_values)

state_values = np.zeros([number_of_values, number_of_values])

x_values = np.sort(x_values)
v_values = np.sort(v_values)

for x in range(len(x_values)):
	for v in range(len(v_values)):
		state = [ x_values[x], v_values[v] ]

		state = agent_obj.scale_state(state)

		state = fourier.compute(state)

		value = 0
		q_vals = []
		greedy_action = []
		best_q = None

		for y in range(len(env_obj.action_list)):
			qval = agent_obj.q_value(state, y)
			q_vals.append(qval)

			if best_q == None or best_q < qval:
				best_q = qval
				greedy_action = [ y ]
			elif best_q == qval:
				greedy_action.append(y)

		qvals = np.array(q_vals)

		prob_dist = (agent_obj.hyperparameters['epsilon']/len(env_obj.action_list))*np.ones(len(env_obj.action_list))
		prob_dist[greedy_action] = prob_dist[greedy_action] + (1 - (agent_obj.hyperparameters['epsilon']/len(env_obj.action_list)))/len(greedy_action)

		value = np.sum(prob_dist*q_vals)

		state_values[x, v] = value

		if x == len(x_values)-1 and v == len(v_values)-1:
			print x_values[x],", ",v_values[v],", ",state_values[x,v]
			# break
	# break

# print state_values

# df = pd.DataFrame({ 'position': x_values, 'velocity': v_values, 'value': state_values })

# print df[:10]

# indices = np.arange(0, len(x_values), show_every)

# x_values = x_values[indices]
# v_values = v_values[indices]

sns.heatmap(state_values, xticklabels=v_values, yticklabels=x_values)

plt.show()