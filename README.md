# Mountain-Car-RL
The mountain car experiment in RL

- Algorithm implementations
* [x] qlearning
* [x] sarsa

## Inferences

- Both algorithms learned pretty quickly.
- Q learning unstable for some ranges of hyperparameters.
- But for the optimal one the variance of returns in later episodes very less.
- Interesting to see value function heatmap.


### Qlearning

- state values range from about -240 to 240!! Interesting note
- very low value region along diagonal - Slight positive gradient in direction of positive velocity.
- overall the plot makes sense.
- But state values coming greater than zero.!!!
- Gradients of state values make sense

### SARSA

- State values range from -360 to 80
- low region concentrated in the corner unlike q value function
- SARSA based learned value functions makes more sense. like assigning higher values to positive velocity in leftmost extreme positions.
- But could be wrong as a negative velocity helps to gain momentum when coming down the slope so that it can climb up the next slope.
- Left end of the position has higher value in SARSA compared to q function

## Inferences continued

- The difference is that SARSA has a higher gradient along the diagonal.
- The ranges learned by the algos are different. Range more for SARSA.
- State Values getting more than 0. !! wrong implementation? :(

TODO : visualize value function change over number of episodes.