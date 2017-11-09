__device__ float scale_input(float x, float xliml, float xlimu)
{
	return (x-xliml)/(xlimu-xliml);
}

__device__ float qvalue(Agent *agent, float *state, int a)
{
	int wl_index = 0;
	int wu_index = 0;
	float sum = 0;
	float be_element = 0;

	switch(a)
	{
		case -1:
			wl_index = 0;
			wu_index = EXPANDED_STATE_SIZE;
			break;

		case 0:
			wl_index = EXPANDED_STATE_SIZE;
			wu_index = 2*EXPANDED_STATE_SIZE;
			break;

		case 1:
			wl_index = 2*EXPANDED_STATE_SIZE;
			wu_index = 3*EXPANDED_STATE_SIZE;
	}

	for(int i=wl_index; i<wu_index; i++)
	{
		be_element = state[i - wl_index];
		sum += agent[threadIdx.x].w[i]*be_element;
	}

	return sum;
}

__global__ void init_agent(Agent *agent)
{
	agent[threadIdx.x].time = 0;
	agent[threadIdx.x].alpha = 0.05;
	agent[threadIdx.x].gamma = 1;
	agent[threadIdx.x].epsilon = 0.5;
	agent[threadIdx.x].be_degree = 1;
}

__global__ void reset_agent(Agent *agent, float *returns)
{
	int i;
	agent[threadIdx.x].time = 0;
	returns[threadIdx.x] = 0;

	for(i=0; i<EXPANDED_STATE_SIZE*NUMBER_OF_ACTIONS; i++)
	{
		agent[threadIdx.x].w[i] = 0;
	}
}

// Set action 0 for cur_action and 1 for next action
__global__ void set_action(Agent *agent, Environment *env, int index, curandState_t* randState)
{
	int i=0;
	int action;
	float dice=0;
	float action_dice = 0;
	int greedy_actions[NUMBER_OF_ACTIONS];
	int action_accumulator = 0;
	float best_q = 0;
	float q_res = 0;
	bool uninitialized = true;

	dice = curand_uniform(&randState[threadIdx.x]);

	if(dice<agent[threadIdx.x].epsilon)
	{
		action_dice = curand_uniform(&randState[threadIdx.x]);
		int temp = action_dice*NUMBER_OF_ACTIONS;
		action = env[threadIdx.x].action_list[temp];
	}

	switch(index){
		case 0:
			if(dice>=agent[threadIdx.x].epsilon)
			{
				for(i=0;i<NUMBER_OF_ACTIONS;i++)
				{
					q_res = qvalue(agent, agent[threadIdx.x].be_cur_state, env[threadIdx.x].action_list[i]);

					if(uninitialized || q_res>best_q)
					{
						uninitialized = false;
						best_q = q_res;
						greedy_actions[0] = env[threadIdx.x].action_list[i];
						action_accumulator = 0;
					}
					else if(q_res==best_q)
					{
						greedy_actions[++action_accumulator] = env[threadIdx.x].action_list[i];
					}
				}

				dice = curand_uniform(&randState[threadIdx.x]);
				
				int temp = (int)(dice*(action_accumulator+1));

				action = greedy_actions[temp];
			}
			agent[threadIdx.x].cur_action = action;
			break;
		case 1:
			if(dice>=agent[threadIdx.x].epsilon)
			{
				for(i=0;i<NUMBER_OF_ACTIONS;i++)
				{
					q_res = qvalue(agent, agent[threadIdx.x].be_next_state, env[threadIdx.x].action_list[i]);

					if(uninitialized || q_res>best_q)
					{
						uninitialized = false;
						best_q = q_res;
						greedy_actions[0] = env[threadIdx.x].action_list[i];
						action_accumulator = 0;
					}
					else if(q_res==best_q)
					{
						greedy_actions[++action_accumulator] = env[threadIdx.x].action_list[i];
					}
				}

				dice = curand_uniform(&randState[threadIdx.x]);

				action = greedy_actions[(int)(dice*(action_accumulator+1))];
			}
			agent[threadIdx.x].next_action = action;
			break;
	}	
}


// Set state 0 for cur_state and 1 for next state
__global__ void set_state(Agent *agent, Environment *env, int index)
{
	float position;
	float velocity;
	float s_position;
	float s_velocity;
	int i,j;
	int be_index=0;

	position = env[threadIdx.x].cur_position;
	velocity = env[threadIdx.x].cur_velocity;

	s_position = scale_input(position, env[threadIdx.x].position_lbound, env[threadIdx.x].position_ubound);
	s_velocity = scale_input(velocity, env[threadIdx.x].velocity_lbound, env[threadIdx.x].velocity_ubound);		

	switch(index){
		case 0:
			// Basis Expansion
			for(i=0; i<agent[threadIdx.x].be_degree+1; i++)
			{
				for(j=0; j<agent[threadIdx.x].be_degree+1; j++)
				{
					agent[threadIdx.x].be_cur_state[be_index] = cosf(i*s_position+j*s_velocity);
					be_index++;
				}
			}

			break;
		case 1:
			
			// Basis Expansion
			for(i=0; i<agent[threadIdx.x].be_degree+1; i++)
			{
				for(j=0; j<agent[threadIdx.x].be_degree+1; j++)
				{
					agent[threadIdx.x].be_next_state[be_index] = cosf(i*s_position+j*s_velocity);
					be_index++;
				}
			}
			break;
	}
}

__global__ void compute_returns(Agent *agent, Environment *env, float *returns)
{
	if(env[threadIdx.x].episode_completion_flag)
	{
		return;
	}
	returns[threadIdx.x] += pow(agent[threadIdx.x].gamma, agent[threadIdx.x].time)*env[threadIdx.x].current_reward;

	// printf("\n%d: %f",threadIdx.x, returns[threadIdx.x]);
}

__global__ void copy_action(Agent *agent)
{
	agent[threadIdx.x].cur_action = agent[threadIdx.x].next_action;
}

// SARSA Update
__global__ void sarsa_update(Agent *agent, Environment *env)
{
	float qsa = 0.0;
	float qsaprime = 0.0;
	float factor = 0.0;
	int wl_index;
	int wu_index;
	int a;
	int i=0;

	qsa = qvalue(agent, agent[threadIdx.x].be_cur_state, agent[threadIdx.x].cur_action);
	qsaprime = qvalue(agent, agent[threadIdx.x].be_next_state, agent[threadIdx.x].next_action);

	if(env[threadIdx.x].episode_completion_flag)
	{
		qsaprime = 0;
	}

	factor = env[threadIdx.x].current_reward + qsaprime - qsa;
	factor *= agent[threadIdx.x].alpha;

	a = agent[threadIdx.x].cur_action;

	switch(a)
	{
		case -1:
			wl_index = 0;
			wu_index = EXPANDED_STATE_SIZE;
			break;

		case 0:
			wl_index = EXPANDED_STATE_SIZE;
			wu_index = 2*EXPANDED_STATE_SIZE;
			break;

		case 1:
			wl_index = 2*EXPANDED_STATE_SIZE;
			wu_index = 3*EXPANDED_STATE_SIZE;
	}

	for(i=wl_index; i<wu_index; i++)
	{
		agent[threadIdx.x].w[i] += factor*agent[threadIdx.x].be_cur_state[i-wl_index];
	}

	// printf("\n");
	// for(i=0; i<EXPANDED_STATE_SIZE*NUMBER_OF_ACTIONS; i++)
	// {
	// 	printf("%f ",agent[threadIdx.x].w[i]);
	// }
}