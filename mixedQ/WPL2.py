from Projection import projection
import random
import numpy as np


def wpl2(prevstate, nstate, action, reward, Q, PI, learning_rate, future_rewards_importance, exploration_rate, eta):

    # Update the Q-table of agent A and agent B
    Q[prevstate][action] = (1 - learning_rate)*Q[prevstate][action] + learning_rate*(reward + future_rewards_importance*max(Q[nstate]))

    for a in range(len(PI[prevstate])):
        difference = 0

        # compute difference between this reward and average reward
        for i in range(len(PI[prevstate])):
            difference += Q[prevstate][a] - Q[prevstate][i]
        difference /= len(PI[prevstate])-1

        # scale to sort of normalize the effect of a policy
        if difference > 0:
            deltaPolicy = 1 - PI[prevstate][a]
        else:
            deltaPolicy = PI[prevstate][a]

        rate = eta * difference * deltaPolicy
        PI[prevstate][a] += rate

    projection(PI[prevstate], exploration_rate)
