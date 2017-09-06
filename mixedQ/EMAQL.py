import numpy as np

def emaql(prevstate, nstate, action, reward, Q, PI, learning_rate, future_rewards_importance,
         phc_learning_rate_win, phc_learning_rate_lose):

    #print(action, Q[prevstate])
    # Update the Q-table of agent A and agent B
    Q[prevstate][action] = (1 - learning_rate)*Q[prevstate][action] + learning_rate*(reward + future_rewards_importance*max(Q[nstate]))

    if action == np.argmax(Q[prevstate]):       # does the selected action by Player A equal to the greedy action?
        vector_1 = np.zeros(len(PI[prevstate]))
        vector_1[action] = 1
        eta = phc_learning_rate_win
        #Policy_A = (1 - eta_winning) * Policy_A + eta_winning * vector_1;
    else:
        vector_1 = np.full((len(PI[prevstate])), 1/(len(PI[prevstate])-1))
        vector_1[action] = 0
        eta = phc_learning_rate_lose
        #Policy_A = (1 - eta_losing) * Policy_A + eta_losing * vector_1;

    for a in range(len(PI[prevstate])):
        PI[prevstate][a] = (1 - eta) * PI[prevstate][a] + eta * vector_1[a]

    #projection(PI[prevstate], exploration_rate)
