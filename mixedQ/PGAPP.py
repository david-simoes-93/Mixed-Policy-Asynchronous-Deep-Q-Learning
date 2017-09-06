from Projection import projection
import numpy as np


def pgaapp(prevstate, nstate, action, reward, Q, PI, learning_rate, future_rewards_importance, eta, gamma):

    num_actions=len(PI[prevstate])

    # Update the Q-table of agent A and agent B
    Q[prevstate][action] = (1 - learning_rate)*Q[prevstate][action] + learning_rate*(reward + future_rewards_importance*max(Q[nstate]))

    Value_A = 0
    for a in range(num_actions):
        Value_A += PI[prevstate][a]*Q[prevstate][a]

    delta_hat_A = [0]*num_actions
    delta_A = [0]*num_actions
    for a in range(num_actions):
        if PI[prevstate][a]==1:
            delta_hat_A[a] = Q[prevstate][a]-Value_A
        else:
            delta_hat_A[a] = (Q[prevstate][a]-Value_A)/(1-PI[prevstate][a])

        delta_A[a] = delta_hat_A[a]-gamma*abs(delta_hat_A[a])*PI[prevstate][a]
        PI[prevstate][a] += eta*delta_A[a]


    projection(PI[prevstate], 0)

    #print(Q[prevstate], PI[prevstate])
