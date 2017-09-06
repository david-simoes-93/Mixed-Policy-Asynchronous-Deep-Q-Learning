import numpy as np
import random


# just a test for Q-learning with a ReplayMemory
def q(prevstate, nstate, action, reward, Q, learning_rate,
          future_rewards_importance, id):

    q.pool[id][q.pool_index[id]][0] = action
    q.pool[id][q.pool_index[id]][1] = reward
    q.pool_index[id] = (q.pool_index[id] + 1) % 500
    q.counter[id] = q.counter[id] + 1

    if q.counter[id]<500:
        return

    indexes = random.sample(range(0, 500), 32)

    for act_i, rew_i in q.pool[id][indexes]:
        # do the TD(eligibility_trace_decay) update
        Q[prevstate][int(act_i)] = (1-learning_rate) * Q[prevstate][int(act_i)] + \
                               learning_rate * (rew_i+future_rewards_importance*max(Q[nstate]))

q.pool = np.zeros([2, 500, 2])
q.pool_index = [0, 0]
q.counter = [0, 0]
