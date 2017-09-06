import random
import matplotlib.pyplot as mpl
import numpy as np
from mixedQ.WPL2 import wpl2


def get_action(prob, s):
    if random.random() < exploration_rate:
        return int(random.random() * number_of_actions)
    action_prob = random.random() * sum(prob[s])
    for i in range(len(prob[s])):
        if action_prob < prob[s][i]:
            return i
        action_prob -= prob[s][i]
    return -1


# converts positons into a unique state number
# eg, agents were at [1,2] [2,3], the state number is 3221
def get_state_number(pos1, pos2):
    # if next to each other
    if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
        return rps_state

    return pos1[0] + pos1[1] * 10 + pos2[0] * 100 + pos2[1] * 1000


def get_action_mod(action):
    if action == 0:
        return [-1, 0]
    if action == 1:
        return [1, 0]
    if action == 2:
        return [0, -1]
    if action == 3:
        return [0, 1]
    return [0, 0]


def update_map_and_positions(pos1, pos2, action1, action2):
    mod1 = get_action_mod(action1)
    mod2 = get_action_mod(action2)

    pos1[0] = (pos1[0] + mod1[0]) % map_size
    pos1[1] = (pos1[1] + mod1[1]) % map_size

    # for player 2 we actually need to check if they're side by side
    new_player2_x = (pos2[0] + mod2[0]) % map_size
    new_player2_y = (pos2[1] + mod2[1]) % map_size
    if new_player2_x != pos1[0] or new_player2_y != pos2[0]:
        pos2[0] = new_player2_x
        pos2[1] = new_player2_y


mpl.show()

number_of_agents = 2
number_of_states = 10000
number_of_actions = 2
exploration_rate = 0.05
learning_rate = .01
future_rewards_importance = 0.9
rps_state = number_of_states - 1

number_of_actions = 4
R1 = [[-1, -1, -1, -1],
      [-1, 2, 1, 3],
      [-1, 3, 2, 1],
      [-1, 1, 3, 2]]  # rock paper scissors
R2 = [[-1, -1, -1, -1],
      [-1, 2, 3, 1],
      [-1, 1, 2, 3],
      [-1, 3, 1, 2]]  # rock paper scissors

plot_prob1 = []
plot_prob2 = []
plot_prob3 = []
plot_prob4 = []
# plot_opp_prob1 = []
# plot_opp_prob2 = []
plot_q0 = []
plot_q1 = []

log_p1 = []
log_p2 = []

map_size = 5
pos1 = [0, 0]
pos2 = [2, 2]

Q1 = np.zeros((number_of_states, number_of_actions)).tolist()
Q2 = np.zeros((number_of_states, number_of_actions)).tolist()

Prob1 = np.full((number_of_states, number_of_actions), 1 / number_of_actions).tolist()
Prob2 = np.full((number_of_states, number_of_actions), 1 / number_of_actions).tolist()

its = []
state = get_state_number(pos1, pos2)
rps_timer = 0
rps_timer_list = []
for iteration in range(50000000):
    #print(pos1, pos2, state)

    action1 = get_action(Prob1, state)
    action2 = get_action(Prob2, state)

    if state != rps_state:
        update_map_and_positions(pos1, pos2, action1, action2)
        reward1 = 0
        reward2 = 0
        rps_timer += 1
    else:
        pos1 = [0, 0]
        pos2 = [2, 2]
        reward1 = R1[action1][action2]
        reward2 = R2[action1][action2]
        rps_timer_list.append(rps_timer)
        rps_timer = 0
    prev_state = state
    state = get_state_number(pos1, pos2)



    # print(0, 0, action1, reward1, Q1, Prob1, Prob_1, C1)
    # print(0, 0, action1, reward1, Q1, Prob1, Prob_1, C1)

    learning_rate = 0.01
    eta = max(0.00001, learning_rate / (100 + iteration / 2000))

    wpl2(prev_state, state, action1, reward1, Q1, Prob1, learning_rate, future_rewards_importance,
         exploration_rate / number_of_actions, eta)
    wpl2(prev_state, state, action2, reward2, Q2, Prob2, learning_rate, future_rewards_importance,
         exploration_rate / number_of_actions, eta)

    if iteration % 1000 == 0:
        its.append(iteration)

        log_p1.append(Prob1[rps_state])
        log_p2.append(Prob2[rps_state])

        plot_prob1.append(Prob1[rps_state][0])
        plot_prob2.append(Prob1[rps_state][1])
        plot_prob3.append(Prob1[rps_state][2])
        plot_prob4.append(Prob1[rps_state][3])

        # plot_opp_prob1.append(Prob2[rps_state][0])
        # plot_opp_prob2.append(Prob2[rps_state][1])

        plot_q0.append(max(Q1[rps_state]))
        plot_q1.append(max(Q2[rps_state]))

        mpl.clf()
        mpl.plot(its, plot_prob1, 'r')
        mpl.plot(its, plot_prob2, 'r--')
        mpl.plot(its, plot_prob3, 'y')
        mpl.plot(its, plot_prob4, 'y--')

        mpl.plot(its, plot_q0)
        mpl.plot(its, plot_q1)
        mpl.pause(0.0001)

    if iteration % 10000 == 0:
        print('1 - ', action1, "%5.2f" % reward1, '[' + ' '.join('{0:.{1}f}'.format(k, 3) for k in Q1[rps_state]) + ']',
              '[' + ' '.join('{0:.{1}f}'.format(k, 3) for k in Prob1[rps_state]) + ']',
              '[' + ' '.join('{0:.{1}f}'.format(k, 3) for k in Prob1[2200]) + ']',np.mean(rps_timer_list))
        print('2 - ', action2, "%5.2f" % reward2, '[' + ' '.join('{0:.{1}f}'.format(k, 3) for k in Q2[rps_state]) + ']',
              '[' + ' '.join('{0:.{1}f}'.format(k, 3) for k in Prob2[rps_state]) + ']',
              '[' + ' '.join('{0:.{1}f}'.format(k, 3) for k in Prob2[2200]) + ']',np.mean(rps_timer_list))
        print()
        rps_timer_list=[]
# input("")


with open("WPL-NRPS.txt", "a") as myfile:
    # print(str(algorithm), str(plot_prob1))
    myfile.write(str(log_p1)+" "+str(log_p2)+" "+its + "\n")
