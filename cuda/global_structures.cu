#define EXPANDED_STATE_SIZE 4
#define NUMBER_OF_ACTIONS 3

struct Environment
{
	int number_of_states;
	int action_list[NUMBER_OF_ACTIONS];
	float terminal_states;
	float position_lbound;
	float position_ubound;
	float velocity_lbound;
	float velocity_ubound;

	bool episode_completion_flag;
	float cur_position;
	float cur_velocity;

	float current_reward;
};

struct State{
	float position;
	float velocity;
};

struct Agent
{
	int time;

	// Hyperparameters
	float alpha;
	float gamma;
	float epsilon;
	int be_degree;

	float be_cur_state[EXPANDED_STATE_SIZE];
	float be_next_state[EXPANDED_STATE_SIZE];
	float w[EXPANDED_STATE_SIZE*NUMBER_OF_ACTIONS];

	int cur_action;
	int next_action;

	bool updation_completion_flag;

};