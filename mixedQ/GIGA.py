from Projection import projection
import numpy as np
import random


def gigawolf(prevstate, nstate, action, reward, Q, PI, Z, learning_rate, future_rewards_importance, eta):

    num_of_actions = len(PI[prevstate])

    # Update the Q-table of agent A and agent B
    #Q[prevstate][action] = (1 - learning_rate)*Q[prevstate][action] + learning_rate*reward
    Q[prevstate][action] = (1 - learning_rate)*Q[prevstate][action] + learning_rate*(reward + future_rewards_importance*max(Q[nstate]))

    # Update the agent's strategy, using the stepsize and *POSSIBLE* rewards
    PI_hat = [0]*num_of_actions
    for a in range(num_of_actions):
        PI_hat[a] = PI[prevstate][a] + eta*Q[prevstate][a]

    # Project this strategy
    projection(PI_hat, 0)

    # Update the agent's 'z' distribution, using the stepsize and *POSSIBLE* rewards
    z = [0]*num_of_actions
    for a in range(num_of_actions):
        z[a] = Z[prevstate][a] + (1/3)*eta*Q[prevstate][a]

    # Project this strategy
    projection(z, 0)

    # Calculate delta using sum of squared differences
    d_num_A = np.sqrt(((np.array(z)-np.array(Z[prevstate]))**2).sum())
    d_denom_A = np.sqrt(((np.array(z)-np.array(PI_hat))**2).sum())
    if d_denom_A == 0:
        delta_A = 1
    else:
        delta_A = min(1, d_num_A/d_denom_A)

    # Do an update of the agent's strategy
    for a in range(num_of_actions):
        Z[prevstate][a]=z[a]
        PI[prevstate][a]=PI_hat[a]+delta_A*(z[a]-PI_hat[a])
