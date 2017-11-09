// #include "global_structures.cu"

__global__ void init_environment(Environment* env){
	env[threadIdx.x].number_of_states = 0;
	
	// Initialize action list
	env[threadIdx.x].action_list[0] = -1;
	env[threadIdx.x].action_list[1] = 0;
	env[threadIdx.x].action_list[2] = 1;

	//Initialize terminal state list
	env[threadIdx.x].terminal_states = 0.5;

	env[threadIdx.x].position_lbound = -1.2;
	env[threadIdx.x].position_ubound = 0.5;
	env[threadIdx.x].velocity_lbound = -0.07;
	env[threadIdx.x].velocity_ubound = 0.07;
}

__global__ void reset_environment(Environment* env){
	env[threadIdx.x].cur_position = -0.5;
	env[threadIdx.x].cur_velocity = 0;
	env[threadIdx.x].episode_completion_flag = false;
}


__global__ void interact(Environment* env, Agent* agent){

	if(env[threadIdx.x].episode_completion_flag)
	{
		return;
	}

	// printf("\npos: %f vel: %f action: %d", env[threadIdx.x].cur_position, env[threadIdx.x].cur_velocity, agent[threadIdx.x].cur_action);

	int action = agent[threadIdx.x].cur_action;

	env[threadIdx.x].cur_velocity += 0.001*action - 0.0025*cosf(3*env[threadIdx.x].cur_position);

	if(env[threadIdx.x].cur_velocity > env[threadIdx.x].velocity_ubound)
	{
		env[threadIdx.x].cur_velocity = env[threadIdx.x].velocity_ubound;
	}
	else{
		if(env[threadIdx.x].cur_velocity < env[threadIdx.x].velocity_lbound)
		{
			env[threadIdx.x].cur_velocity = env[threadIdx.x].velocity_lbound;
		}
	}

	env[threadIdx.x].cur_position += env[threadIdx.x].cur_velocity;

	if(env[threadIdx.x].cur_position > env[threadIdx.x].position_ubound)
	{
		env[threadIdx.x].episode_completion_flag = true;
		env[threadIdx.x].cur_position = env[threadIdx.x].position_ubound;
	}
	else if(env[threadIdx.x].cur_position < env[threadIdx.x].position_lbound)
	{
		env[threadIdx.x].cur_velocity = 0;
		env[threadIdx.x].cur_position = env[threadIdx.x].position_lbound;
	}

	env[threadIdx.x].current_reward = -1;
}