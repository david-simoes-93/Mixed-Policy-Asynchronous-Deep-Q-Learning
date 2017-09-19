import numpy as np
from mixedQ.Projection import projection


## WOLF_PHC
def wolf_phc(best_action, a_size, curr_policy, curr_policy_slow, c, curr_q):
    target_policy = np.zeros([len(best_action), a_size])
    target_policy_slow = np.zeros([len(best_action), a_size])

    for index in range(len(best_action)):
        # update average policy estimate
        for i in range(a_size):
            target_policy_slow[index][i] = curr_policy_slow[index][i] + \
                                           (curr_policy[index][i] - curr_policy_slow[index][i]) / c

        # find out whether winning or losing, and set delta correspondingly
        sum1 = 0
        sum2 = 0
        for i in range(a_size):
            sum1 += curr_policy[index][i] * curr_q[index][i]
            sum2 += target_policy_slow[index][i] * curr_q[index][i]
        winning = sum1 > sum2
        if winning:
            d = 1 / 200
        else:
            d = 1 / 100

        # each subopt action is penalized by at most delta/#(subopt)
        for a in range(a_size):
            target_policy[index][a] = curr_policy[index][a]

        for a in range(a_size):
            if a != best_action[index]:
                target_policy[index][a] -= d / (a_size - 1)
            else:
                target_policy[index][a] += d

            if target_policy[index][a] < 0:
                target_policy[index][best_action[index]] += target_policy[index][a]
                target_policy[index][a] = 0
                # """

    return target_policy, target_policy_slow

##GIGA_WOLF
def giga_wolf(best_action, a_size, curr_policy, curr_policy_slow, curr_q):
    target_policy = np.zeros([len(best_action), a_size])
    target_policy_slow = np.zeros([len(best_action), a_size])

    for index in range(len(best_action)):
        # Update the agent's strategy, using the stepsize and *POSSIBLE* rewards
        PI_hat = [0] * a_size
        for a in range(a_size):
            PI_hat[a] = curr_policy[index][a] + (1 / 100) * curr_q[index][a]

        # Project this strategy
        projection(PI_hat, 0)

        # Update the agent's 'z' distribution, using the stepsize and *POSSIBLE* rewards
        z = [0] * a_size
        for a in range(a_size):
            z[a] = curr_policy_slow[index][a] + (1 / 300) * curr_q[index][a]

        # Project this strategy
        projection(z, 0)

        # Calculate delta using sum of squared differences
        d_num_A = np.sqrt(((np.array(z) - np.array(curr_policy_slow[index])) ** 2).sum())
        d_denom_A = np.sqrt(((np.array(z) - np.array(PI_hat)) ** 2).sum())
        if d_denom_A == 0:
            delta_A = 1
        else:
            delta_A = min(1, d_num_A / d_denom_A)

        # Do an update of the agent's strategy
        for a in range(a_size):
            target_policy_slow[index][a] = z[a]
            target_policy[index][a] = PI_hat[a] + delta_A * (z[a] - PI_hat[a])
            # """

    return target_policy, target_policy_slow

## EMA-QL
def ema_ql(best_action, a_size, curr_policy, actions):
    target_policy = np.zeros([len(best_action), a_size])

    for index in range(len(best_action)):
        if actions[index] == best_action[index]:  # does the selected action by Player A equal to the greedy action?
            vector_1 = np.zeros(a_size)
            vector_1[actions[index]] = 1
            eta = 1 / 200
            # Policy_A = (1 - eta_winning) * Policy_A + eta_winning * vector_1;
        else:
            vector_1 = np.full(a_size, 1 / (a_size - 1))
            vector_1[actions[index]] = 0
            eta = 1 / 100
            # Policy_A = (1 - eta_losing) * Policy_A + eta_losing * vector_1;

        # print((1 - eta) * curr_policy[index] + eta * vector_1)
        target_policy[index] = (1 - eta) * curr_policy[index] + eta * vector_1

    return target_policy

## WPL
def wpl(best_action, a_size, curr_policy, curr_q):
    target_policy = np.zeros([len(best_action), a_size])

    for index in range(len(best_action)):
        for a in range(a_size):
            difference = 0

            # compute difference between this reward and average reward
            for i in range(a_size):
                difference += curr_q[index][a] - curr_q[index][i]
            difference /= a_size - 1

            # scale to sort of normalize the effect of a policy
            if difference > 0:
                deltaPolicy = 1 - curr_policy[index][a]
            else:
                deltaPolicy = curr_policy[index][a]

            rate = 1 / 100 * difference * deltaPolicy
            # print(difference, Q[prevstate], eta, difference, deltaPolicy, rate)
            target_policy[index][a] = curr_policy[index][a] + rate

        projection(target_policy[index], 0.05/a_size)

    return target_policy

##PGA-APP
def pag_app(best_action, a_size, curr_policy, curr_q):
    target_policy = np.zeros([len(best_action), a_size])

    for index in range(len(best_action)):
        Value_A = 0
        for a in range(a_size):
            Value_A += curr_policy[index][a] * curr_q[index][a]

        delta_hat_A = [0] * a_size
        delta_A = [0] * a_size
        for a in range(a_size):
            if curr_policy[index][a] == 1:
                delta_hat_A[a] = curr_q[index][a] - Value_A
            else:
                delta_hat_A[a] = (curr_q[index][a] - Value_A) / (1 - curr_policy[index][a])

            delta_A[a] = delta_hat_A[a] - 1 * abs(delta_hat_A[a]) * curr_policy[index][a]
            target_policy[index][a] = curr_policy[index][a] + 1 / 100 * delta_A[a]

        projection(target_policy[index], 0.05/a_size)

    return target_policy