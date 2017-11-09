from environment import Environment
from agent import Agent
from agent import FourierExpansion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mode = "s"

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

x_values = (env_obj.position_ubound-env_obj.position_lbound)*np.random.sample(number_of_values) + env_obj.position_lbound
v_values = (env_obj.velocity_ubound-env_obj.velocity_lbound)*np.random.sample(number_of_values) + env_obj.velocity_lbound
state_values = np.zeros([number_of_values, number_of_values])

x_values = np.sort(x_values)
v_values = np.sort(v_values)

for x in range(len(x_values)):
	for v in range(len(v_values)):
		state = [ x_values[x], v_values[v] ]

		state = agent_obj.scale_state(state)

		state = fourier.compute(state)

		value = 0

		for y in range(len(env_obj.action_list)):
			value += agent_obj.q_value(state, y)

		state_values[x, v] = value

# print state_values

# df = pd.DataFrame({ 'position': x_values, 'velocity': v_values, 'value': state_values })

# print df[:10]

# indices = np.arange(0, len(x_values), show_every)

# x_values = x_values[indices]
# v_values = v_values[indices]

sns.heatmap(state_values, xticklabels=v_values, yticklabels=x_values)

plt.show()