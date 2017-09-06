import numpy as np
import random


def wolfq(prevstate, nstate, action, reward, Q, PI, PIavg, C, learning_rate,
          future_rewards_importance, phc_learning_rate_win, phc_learning_rate_lose):
    number_of_actions = len(Q[prevstate])

    # do the TD(eligibility_trace_decay) update
    Q[prevstate][action] = (1-learning_rate) * Q[prevstate][action] + \
                           learning_rate * (reward+future_rewards_importance*max(Q[nstate]))

    # increment the state counter
    # Ci(s) = Ci(s) + 1
    C[prevstate] += 1

    # update average policy estimate
    for i in range(len(PI[prevstate])):
        PIavg[prevstate][i] += (PI[prevstate][i] - PIavg[prevstate][i]) / C[prevstate]

    # find out whether winning or losing, and set delta correspondingly
    sum1=0
    sum2=0
    for i in range(len(PI[prevstate])):
        sum1 += PI[prevstate][i] * Q[prevstate][i]
        sum2 += PIavg[prevstate][i] * Q[prevstate][i]
    winning = sum1 > sum2
    if winning:
        d = phc_learning_rate_win
    else:
        d = phc_learning_rate_lose

    # find out which actions would now be optimal
    opt = argmax(Q, prevstate)
    #print('opt', Q, opt)

    # each subopt action is penalized by at most delta/#(subopt)
    for a in range(len(PI[prevstate])):
        if a not in opt:
            PI[prevstate][a] -= d / (number_of_actions - len(opt))
        else:
            PI[prevstate][a] += d/len(opt)

        if PI[prevstate][a] < 0:
            for o in opt:
                PI[prevstate][o] += PI[prevstate][a]/len(opt)
            PI[prevstate][a] = 0


def argmax(Q, s):
    maxval = Q[s][0]
    # ret_val=0
    for i in range(1, len(Q[s])):
        if Q[s][i] > maxval:
            # ret_val=i
            maxval = Q[s][i]
    l = []
    for i in range(0, len(Q[s])):
        if Q[s][i] == maxval:
            l.append(i)
    return l  # ret_val

