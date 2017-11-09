#include <unistd.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "global_structures.cu"
#include "environment.cu"
#include "agent.cu"

#define NUMBER_OF_THREADS 2
#define NUMBER_OF_TRIALS 2
#define NUMBER_OF_EPISODES 2
#define HORIZON 20

// Environment variables

/* this GPU kernel function is used to initialize the random states */
__global__ void init_random_states(unsigned int seed, curandState_t* states) {

	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
	          threadIdx.x, /* the sequence number should be different for each core (unless you want all
	                         cores to get the same sequence of numbers for some reason - use thread id! */
	          0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	          &states[threadIdx.x]);
}

void interface(){
	// Initialize and copy environment variables to device

	int environment_size = sizeof(Environment);
	int agent_size = sizeof(Agent);
	int number_of_local_threads = NUMBER_OF_THREADS;
	int trial=0;
	int number_of_steps = 0;
	int episode = 0;
	int i, j;

	// struct environment env[NUMBER_OF_THREADS];
	struct Environment *d_env;
	struct Agent *d_agent;
	float *d_returns;

	float returns[NUMBER_OF_TRIALS][NUMBER_OF_EPISODES];

	curandState_t* random_states;

	for(trial=0; trial<NUMBER_OF_TRIALS; trial+=NUMBER_OF_THREADS)
	{
		printf("\nCompleted Trials %d/%d",trial, NUMBER_OF_TRIALS);
		
		if(trial+NUMBER_OF_THREADS > NUMBER_OF_TRIALS)
		{
			number_of_local_threads = NUMBER_OF_TRIALS % NUMBER_OF_THREADS;
		}

		printf("\nNumber of parallel trials: %d", number_of_local_threads);

		float h_returns[number_of_local_threads];

		// Allocate memory to enivronment variables
		cudaMalloc((void**)&d_env, number_of_local_threads*environment_size);

		// Allocate memory to agent variables
		cudaMalloc((void**)&d_agent, number_of_local_threads*agent_size);

		// Allocate memory to random state variable
		cudaMalloc((void**) &random_states, number_of_local_threads * sizeof(curandState_t));

		//Allocate memory for returns storage
		cudaMalloc((void**) &d_returns, number_of_local_threads * sizeof(float));		

		// Start Computations

		init_random_states<<<1, number_of_local_threads>>>(time(0), random_states);
		init_environment<<<1, number_of_local_threads>>>(d_env);
		init_agent<<<1, number_of_local_threads>>>(d_agent);

		for(episode=0; episode<NUMBER_OF_EPISODES; episode++)
		{
			printf("\nStarting Episode %d \r",episode);
			// printf("\b\b\b\b\b# %3d%%", episode);

			reset_environment<<<1, number_of_local_threads>>>(d_env);
			reset_agent<<<1, number_of_local_threads>>>(d_agent, d_returns);

			for(number_of_steps=0; number_of_steps<HORIZON; number_of_steps++)
			{
				if(number_of_steps == 0)
				{
					set_state<<<1, number_of_local_threads>>>(d_agent, d_env, 0);
					set_action<<<1, number_of_local_threads>>>(d_agent, d_env, 0, random_states);
				}

				interact<<<1, number_of_local_threads>>>(d_env, d_agent);

				set_state<<<1, number_of_local_threads>>>(d_agent, d_env, 1);
				set_action<<<1, number_of_local_threads>>>(d_agent, d_env, 1, random_states);

				compute_returns<<<1, number_of_local_threads>>>(d_agent, d_env, d_returns);

				sarsa_update<<<1, number_of_local_threads>>>(d_agent, d_env);

				set_state<<<1, number_of_local_threads>>>(d_agent, d_env, 0);
				copy_action<<<1, number_of_local_threads>>>(d_agent);
			}

			printf("\nCopying from device to host");
			cudaMemcpy(h_returns, d_returns, number_of_local_threads*sizeof(float), cudaMemcpyDeviceToHost);

			printf("\n Saving");	
			for(i=0; i<number_of_local_threads; i++)
			{
				printf("\nreturns: %f", h_returns[i]);
				returns[trial+i][episode] = h_returns[i];
			}
		}

		FILE *f = fopen("returns.txt", "w");
		// fwrite(returns, sizeof(float), sizeof(returns), f);
		
		for(i=0; i<NUMBER_OF_TRIALS; i++)
		{
			for(j=0; j< NUMBER_OF_EPISODES; j++)
			{
				fprintf(f, "%f;", returns[i][j]);
			}
			fprintf(f, "\n");
		}

		fclose(f);

		printf("End return history");

		// End Computations
		
		// Free variables
		cudaFree(d_env);
		cudaFree(d_agent);
		cudaFree(random_states);
		cudaFree(d_returns);
	}

	printf("\nExecution Complete");
}


int main(void)
{
	interface();
	return 0;
}