from time import time
import tensorflow as tf
from asyncdqn.QNetwork1step import QNetwork1Step
from asyncdqn.QNetworkPolicy import QNetworkPolicy
from asyncdqn.QNetworkPolicySlow import QNetworkPolicySlow
from asyncdqn.Helper import update_target_graph, discount, make_gif, get_empty_loss_arrays
import numpy as np
from mixedQ.Projection import projection
from time import sleep
import math


# Worker class
class WorkerRandomRPS:
    def __init__(self, game, name, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                 width=15, height=15, number_of_cell_types=3, use_lstm=False, use_conv_layers=False, save_gifs=False,
                 number_of_predators=5, number_of_prey=5, type_of_algorithm=1):
        self.name = "worker_" + str(name)
        self.is_chief = self.name == 'worker_0'
        print(self.name)
        self.number = name
        self.model_path = model_path
        self.trainer_predator = trainer_predator
        self.trainer_prey = trainer_prey
        self.global_episodes = global_episodes
        self.save_gifs = save_gifs

        self.episode_rewards_predator = []
        self.episode_rewards_prey = []
        self.episode_lengths = []
        self.episode_mean_values_predator = []
        self.episode_mean_values_prey = []
        with tf.variable_scope(self.name):
            self.increment = self.global_episodes.assign_add(1)
            self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.type_of_algorithm = type_of_algorithm
        if type_of_algorithm in [3, 4, 5]:
            self.local_AC_predator = QNetworkPolicy(s_size, a_size, self.name + "_predator", trainer_predator,
                                                    use_conv_layers, use_lstm, width, height, number_of_cell_types)
            self.local_AC_prey = QNetworkPolicy(s_size, a_size, self.name + "_prey", trainer_prey, use_conv_layers,
                                                use_lstm,
                                                width, height, number_of_cell_types)
        elif type_of_algorithm in [1, 2]:
            self.local_AC_predator = QNetworkPolicySlow(s_size, a_size, self.name + "_predator", trainer_predator,
                                                        use_conv_layers, use_lstm, width, height, number_of_cell_types)
            self.local_AC_prey = QNetworkPolicySlow(s_size, a_size, self.name + "_prey", trainer_prey, use_conv_layers,
                                                    use_lstm,
                                                    width, height, number_of_cell_types)
        else:
            self.local_AC_predator = QNetwork1Step(s_size, a_size, self.name + "_predator", trainer_predator,
                                                   use_conv_layers, use_lstm, width, height, number_of_cell_types)
            self.local_AC_prey = QNetwork1Step(s_size, a_size, self.name + "_prey", trainer_prey, use_conv_layers,
                                               use_lstm, width, height, number_of_cell_types)
        self.update_local_ops_predator = update_target_graph('global_predator', self.name + "_predator")
        self.update_local_ops_prey = update_target_graph('global_prey', self.name + "_prey")

        # Env Pursuit set-up
        self.actions = np.identity(a_size, dtype=bool).tolist()
        self.env = game

        self.height = height
        self.width = width
        self.number_of_cell_types = number_of_cell_types
        self.use_lstm = use_lstm
        self.s_size = s_size
        self.use_conv_layers = use_conv_layers
        self.number_of_predators = number_of_predators
        self.number_of_prey = number_of_prey
        self.a_size = a_size

        self.exploration_rate_predators = 0
        self.exploration_rate_prey = 0

        self.c = 1

        self.policy_evolution = []

    def print_screen(self, screen):
        m = np.reshape(screen, [self.height, self.width])
        for row in range(self.height):
            for column in range(self.width):
                print(int(m[column][row] * self.number_of_cell_types + 0.01), end=' ')
            print()

    def print_screen_formatted(self, screen):
        m = np.reshape(screen, [self.height, self.width, self.number_of_cell_types])
        for row in range(self.height):
            for column in range(self.width):
                for c in range(self.number_of_cell_types):
                    print(int(m[column][row][c]), end='')
                print('', end=' ')
            print()

    def train(self, rollout, sess, gamma, ac_network):
        # print(self.number, "Train")
        """rollout.append([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                       1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0.], 3, 1, None, True, 0])"""

        rollout = np.array(rollout)
        observations = rollout[:, 0]  # state t
        actions = rollout[:, 1]  # action taken at timestep t
        rewards = rollout[:, 2]  # reward t
        next_observations = np.vstack(rollout[:, 3])  # state t+1
        terminals = rollout[:, 4]  # whether timestep t was terminal

        next_max_q = rollout[:, 5]  # the best Q value of state t+1 as calculated by target network
        best_action = rollout[:, 6]
        curr_policy = rollout[:, 7]
        curr_q = rollout[:, 8]  # not from target
        curr_policy_slow = rollout[:, 9]

        target_policy = np.zeros([len(best_action), self.a_size])
        target_policy_slow = np.zeros([len(best_action), self.a_size])

        self.c += 1

        if self.type_of_algorithm == 1:
            ## WOLF_PHC
            for index in range(len(best_action)):
                # update average policy estimate
                for i in range(self.a_size):
                    target_policy_slow[index][i] = curr_policy_slow[index][i] + (curr_policy[index][i] -
                                                                                 curr_policy_slow[index][
                                                                                     i]) / self.c  # todo C[prevstate]

                # find out whether winning or losing, and set delta correspondingly
                sum1 = 0
                sum2 = 0
                for i in range(self.a_size):
                    sum1 += curr_policy[index][i] * curr_q[index][i]
                    sum2 += target_policy_slow[index][i] * curr_q[index][i]
                winning = sum1 > sum2
                if winning:
                    d = 1 / 200
                else:
                    d = 1 / 100

                # each subopt action is penalized by at most delta/#(subopt)
                for a in range(self.a_size):
                    target_policy[index][a] = curr_policy[index][a]

                for a in range(self.a_size):
                    if a != best_action[index]:
                        target_policy[index][a] -= d / (self.a_size - 1)
                    else:
                        target_policy[index][a] += d

                    if target_policy[index][a] < 0:
                        target_policy[index][best_action[index]] += target_policy[index][a]
                        target_policy[index][a] = 0
                        # """
        elif self.type_of_algorithm == 2:
            ##GIGA_WOLF
            for index in range(len(best_action)):
                # Update the agent's strategy, using the stepsize and *POSSIBLE* rewards
                PI_hat = [0] * self.a_size
                for a in range(self.a_size):
                    PI_hat[a] = curr_policy[index][a] + (1 / 100) * curr_q[index][a]

                # Project this strategy
                projection(PI_hat, 0)

                # Update the agent's 'z' distribution, using the stepsize and *POSSIBLE* rewards
                z = [0] * self.a_size
                for a in range(self.a_size):
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
                for a in range(self.a_size):
                    target_policy_slow[index][a] = z[a]
                    target_policy[index][a] = PI_hat[a] + delta_A * (z[a] - PI_hat[a])
                    # """
        elif self.type_of_algorithm == 3:
            ## EMA-QL
            for index in range(len(best_action)):
                if actions[index] == best_action[
                    index]:  # does the selected action by Player A equal to the greedy action?
                    vector_1 = np.zeros(self.a_size)
                    vector_1[actions[index]] = 1
                    eta = 1 / 200
                    # Policy_A = (1 - eta_winning) * Policy_A + eta_winning * vector_1;
                else:
                    vector_1 = np.full(self.a_size, 1 / (self.a_size - 1))
                    vector_1[actions[index]] = 0
                    eta = 1 / 100
                    # Policy_A = (1 - eta_losing) * Policy_A + eta_losing * vector_1;
                # print((1 - eta) * curr_policy[index] + eta * vector_1)
                target_policy[index] = (1 - eta) * curr_policy[index] + eta * vector_1
                # projection(target_policy[index], 0.05)
                # """
        elif self.type_of_algorithm == 4:
            ## WPL
            for index in range(len(best_action)):

                for a in range(self.a_size):
                    difference = 0

                    # compute difference between this reward and average reward
                    for i in range(self.a_size):
                        difference += curr_q[index][a] - curr_q[index][i]
                    difference /= self.a_size - 1

                    # scale to sort of normalize the effect of a policy
                    if difference > 0:
                        deltaPolicy = 1 - curr_policy[index][a]
                    else:
                        deltaPolicy = curr_policy[index][a]

                    rate = 1 / 100 * difference * deltaPolicy
                    # print(difference, Q[prevstate], eta, difference, deltaPolicy, rate)
                    target_policy[index][a] = curr_policy[index][a] + rate

                projection(target_policy[index], 0.05)
                # """
        elif self.type_of_algorithm == 5:
            ##PGAAPP
            for index in range(len(best_action)):
                Value_A = 0
                for a in range(self.a_size):
                    Value_A += curr_policy[index][a] * curr_q[index][a]

                delta_hat_A = [0] * self.a_size
                delta_A = [0] * self.a_size
                for a in range(self.a_size):
                    if curr_policy[index][a] == 1:
                        delta_hat_A[a] = curr_q[index][a] - Value_A
                    else:
                        delta_hat_A[a] = (curr_q[index][a] - Value_A) / (1 - curr_policy[index][a])

                    delta_A[a] = delta_hat_A[a] - 1 * abs(delta_hat_A[a]) * curr_policy[index][a]
                    target_policy[index][a] = curr_policy[index][a] + 1 / 100 * delta_A[a]

                projection(target_policy[index], 0.05)
                # """

        # we get rewards, terminals, prev_screen, next screen, and target network
        discounted_rewards = (1. - terminals) * gamma * next_max_q + rewards

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        if self.use_lstm:
            rnn_state = ac_network.state_init
            feed_dict = {ac_network.target_q_t: discounted_rewards,
                         ac_network.inputs: np.vstack(observations),
                         ac_network.actions: actions,
                         ac_network.state_in[0]: rnn_state[0],
                         ac_network.state_in[1]: rnn_state[1],
                         ac_network.target_policy_t: target_policy}
        else:
            feed_dict = {ac_network.target_q_t: discounted_rewards,
                         # ac_network.computed_q_t: values,
                         ac_network.network.inputs: np.vstack(observations),
                         ac_network.actions: actions,
                         ac_network.target_policy_t: target_policy,
                         ac_network.target_policy_slow_t: target_policy_slow}
        v_l, q_value, q_acted, g_n, v_n, _ = sess.run(
            [ac_network.loss, ac_network.network.value, ac_network.q_acted,
             ac_network.grad_norms,
             ac_network.var_norms,
             ac_network.apply_grads],
            feed_dict=feed_dict)

        # print(debug1, target_policy)
        # print(debug3[0], " ", debug1[0], debug2[0])
        # v_l, q_value, q_acted, g_n, v_n = 0,0,0,0,0

        """index = len(observations) - 1
        self.print_screen_formatted(observations[index])
        print(actions[index], rewards[index], terminals[index])
        #self.print_screen_formatted(next_observations[index])
        print(next_max_q[index], discounted_rewards[index])
        print(v_l, q_value[index], q_acted[index])
        #input("continue?")"""

        return v_l / len(rollout), 0, 0, g_n, v_n

    # Take an action using probabilities from policy network output.
    def take_action_from_network(self, sess, network, number_of_agents, previous_screen, action_indexes,
                                 action_distribution, value, rnn_state):
        action = [0] * number_of_agents

        if self.use_lstm:
            for pred_index in range(number_of_agents):
                action_distribution[pred_index], value[pred_index], rnn_state[
                    pred_index] = sess.run(
                    [network.policy, network.value, network.state_out],
                    feed_dict={network.inputs: [previous_screen[pred_index]],
                               network.state_in[0]: rnn_state[pred_index][0],
                               network.state_in[1]: rnn_state[pred_index][1]})
        else:
            action_distribution, value = sess.run(
                [network.policy, network.value],
                feed_dict={network.inputs: previous_screen})

        for pred_index in range(number_of_agents):
            action[pred_index] = np.random.choice(action_indexes, p=action_distribution[pred_index])

        return action, action_distribution, value, rnn_state

    def work(self, max_episode_length, gamma, sess, coord=None, saver=None):
        episode_count = 0  # sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        # with sess.as_default(), sess.graph.as_default():
        prev_clock = time()
        if coord is None:
            coord = sess

        sess.run([self.local_AC_predator.assign_op, self.local_AC_prey.assign_op])
        print(self.number, "Resetting target network")

        action_indexes = list(range(self.env.numberOfActions))

        # self.env.new_episode()
        previous_screen_predator, previous_screen_prey, r0, r1 = self.env.observe()  # .screen_buffer
        # episode_frames.append(self.env.birdseye_observe())

        # Initial state; initialize values for the LSTM network
        action_predator, action_distribution_predator, value_predator, rnn_state_predator = \
            self.take_action_from_network(sess, self.local_AC_predator.network, self.number_of_predators,
                                          previous_screen_predator, action_indexes,
                                          [], [], [])
        action_prey, action_distribution_prey, value_prey, rnn_state_prey = \
            self.take_action_from_network(sess, self.local_AC_prey.network, self.number_of_prey,
                                          previous_screen_prey, action_indexes,
                                          [], [], [])

        # prev_vals = sess.run(self.local_AC_predator.local_vars)
        while not coord.should_stop():
            end_epsilon_probs = [0.01, 0.1, 0.5]
            self.exploration_rate_predators -= (1.0 - end_epsilon_probs[self.number % 3]) / 5000000.0
            if self.exploration_rate_predators < end_epsilon_probs[self.number % 3]:
                self.exploration_rate_predators = end_epsilon_probs[self.number % 3]

            # print("Copying global networks to local networks")
            sess.run(self.update_local_ops_predator)
            sess.run(self.update_local_ops_prey)

            """if self.is_chief and episode_count % 100 == 0:
                sess.run(self.local_AC_predator.assign_op)
                sess.run(self.local_AC_prey.assign_op)
                print("Updating target network")"""

            episode_buffer_predator = []
            episode_values_predator = []
            for pred_index in range(self.number_of_predators):
                episode_buffer_predator.append([])
                episode_values_predator.append([])
            episode_buffer_prey = []
            episode_values_prey = []
            for prey_index in range(self.number_of_prey):
                episode_buffer_prey.append([])
                episode_values_prey.append([])
            episode_frames = []
            episode_reward_predator = 0
            episode_reward_prey = 0
            episode_step_count = 0

            batch_size = 25
            for step in range(batch_size):
                # input("new turn")
                # print("wat")
                # sleep(1)

                # times_measurement = []
                v_l_prey, p_l_prey, e_l_prey, g_n_prey, v_n_prey = get_empty_loss_arrays(self.number_of_prey)
                v_l, p_l, e_l, g_n, v_n = get_empty_loss_arrays(self.number_of_predators)

                # print(np.floor(previous_screen_predator))
                # print(np.floor(previous_screen_prey))
                # act predator
                if np.random.random() < self.exploration_rate_predators:
                    action_predator = np.random.choice(action_indexes, self.number_of_predators)
                    # Otherwise, lets judge and execute a move
                else:
                    action_predator, action_distribution_predator, value_predator, rnn_state_predator = \
                        self.take_action_from_network(sess, self.local_AC_predator.network, self.number_of_predators,
                                                      previous_screen_predator, action_indexes,
                                                      action_distribution_predator, value_predator, rnn_state_predator)
                    # print("value ", value_predator)

                # act prey
                if np.random.random() < self.exploration_rate_prey:
                    action_prey = np.random.choice(action_indexes, self.number_of_prey)
                else:
                    action_prey, action_distribution_prey, value_prey, rnn_state_prey = \
                        self.take_action_from_network(sess, self.local_AC_prey.network, self.number_of_prey,
                                                      previous_screen_prey, action_indexes,
                                                      action_distribution_prey, value_prey, rnn_state_prey)

                self.env.act([action_predator[0], action_prey[0]])

                # Watch environment
                current_screen_predator, current_screen_prey, reward_pred, reward_prey = self.env.observe()
                # print(reward_pred, reward_prey,current_screen_predator, current_screen_prey)
                # print(action_predator[0], action_prey[0],reward_pred, reward_prey)
                # print(np.floor(current_screen_predator))
                # print(np.floor(current_screen_prey))
                # print("------")
                # sleep(1)
                # current_screen_predator, reward, terminal_predator = self.env.predator_observe()
                # reward = np.sum(reward)
                episode_reward_predator += reward_pred[0]
                episode_reward_prey += reward_prey[0]

                episode_frames.append(self.env.birdseye_observe())

                # get target network values
                next_max_q = sess.run([self.local_AC_predator.target_network.best_q],
                                      feed_dict={self.local_AC_predator.target_network.inputs: current_screen_predator})
                best_action, policy, policy_slow, value_predator = sess.run(
                    [self.local_AC_predator.target_network.best_action_index,
                     self.local_AC_predator.target_network.policy,
                     self.local_AC_predator.target_network.policy_slow,
                     self.local_AC_predator.target_network.value],
                    feed_dict={self.local_AC_predator.target_network.inputs: previous_screen_predator})

                # Store environment
                for pred_index in range(self.number_of_predators):
                    episode_buffer_predator[pred_index].append(
                        [previous_screen_predator[pred_index], action_predator[pred_index], reward_pred,
                         current_screen_predator[pred_index], False, next_max_q[pred_index],
                         best_action[pred_index], policy[pred_index], value_predator[pred_index],
                         policy_slow[pred_index]])
                    episode_values_predator[pred_index].append(np.max(value_predator[pred_index]))
                previous_screen_predator = current_screen_predator

                # get target network values
                next_max_q_prey = sess.run([self.local_AC_prey.target_network.best_q],
                                           feed_dict={self.local_AC_prey.target_network.inputs: current_screen_prey})
                best_action, policy, policy_slow, value_prey = sess.run(
                    [self.local_AC_prey.target_network.best_action_index,
                     self.local_AC_prey.target_network.policy,
                     self.local_AC_prey.target_network.policy_slow,
                     self.local_AC_prey.target_network.value],
                    feed_dict={self.local_AC_prey.target_network.inputs: previous_screen_prey})

                # Store environment
                for prey_index in range(self.number_of_prey):
                    episode_buffer_prey[prey_index].append(
                        [previous_screen_prey[prey_index], action_prey[prey_index], reward_prey,
                         current_screen_prey[prey_index], False, next_max_q_prey[prey_index],
                         best_action[prey_index], policy[prey_index], value_prey[prey_index],
                         policy_slow[prey_index]])
                    episode_values_prey[prey_index].append(np.max(value_prey[prey_index]))
                previous_screen_prey = current_screen_prey

                # Measure time and increase episode step count
                total_steps += 1
                if total_steps % 2000 == 0:
                    new_clock = time()
                    # self.speed_up = np.mean(times_measurement)
                    print(2000.0 / (new_clock - prev_clock), "it/s,   ")  # , self.speed_up,"speed-up")
                    prev_clock = new_clock
                    # times_measurement=[]
                episode_step_count += 1

                # print("0ver ",episode_count)
                if reward_pred[0] != 0:
                    self.episode_rewards_predator.append(episode_reward_predator)
                    self.episode_rewards_prey.append(episode_reward_prey)
                    self.episode_lengths.append(episode_step_count)
                    episode_step_count=0

                    self.episode_mean_values_predator.append(np.mean(episode_values_predator))
                    self.episode_mean_values_prey.append(np.mean(episode_values_prey))

            # Update the network using the experience buffer at the end of the episode.
            for pred_index in range(self.number_of_predators):
                if len(episode_buffer_predator[pred_index]) != 0:
                    v_l[pred_index], p_l[pred_index], e_l[pred_index], g_n[pred_index], v_n[pred_index] = \
                        self.train(episode_buffer_predator[pred_index], sess, gamma, self.local_AC_predator)

            # Update the network using the experience buffer at the end of the episode.
            for prey_index in range(self.number_of_prey):
                if len(episode_buffer_prey[prey_index]) != 0:
                    # print(len(episode_buffer_prey[prey_index]), episode_buffer_prey[prey_index])
                    v_l_prey[prey_index], p_l_prey[prey_index], e_l_prey[prey_index], g_n_prey[prey_index], \
                    v_n_prey[prey_index] = \
                        self.train(episode_buffer_prey[prey_index], sess, gamma, self.local_AC_prey)

            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if episode_count % 5 == 0 and episode_count != 0:
                # Save current model
                if self.is_chief and episode_count % 250 == 0:
                    if saver is not None:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")
                    pol1, pol2, q1, q2 = sess.run(
                        [self.local_AC_predator.network.policy, self.local_AC_prey.network.policy,
                         self.local_AC_predator.network.value, self.local_AC_prey.network.value],
                        feed_dict={self.local_AC_predator.network.inputs: [[-1] * 25],
                                   self.local_AC_prey.network.inputs: [[-1] * 25]})
                    self.policy_evolution.append([list(pol1[0]), list(pol2[0]), list(q1[0]), list(q2[0]), self.episode_lengths[-1]])
                    print(self.policy_evolution[-1])

                # Save statistics for TensorBoard
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_reward_predator = np.mean(self.episode_rewards_predator[-5:])
                mean_value_predator = np.mean(self.episode_mean_values_predator[-5:])
                mean_reward_prey = np.mean(self.episode_rewards_prey[-5:])
                mean_value_prey = np.mean(self.episode_mean_values_prey[-5:])

                if episode_count % 500 == 0:
                    summary = tf.Summary()

                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))  # avg episode length

                    summary.value.add(tag='Perf/Reward Predator',
                                      simple_value=float(mean_reward_predator))  # avg reward
                    summary.value.add(tag='Perf/Value Predator',
                                      simple_value=float(mean_value_predator))  # avg episode value_predator
                    summary.value.add(tag='Losses/Value Loss Predator', simple_value=float(np.mean(v_l)))  # value_loss
                    summary.value.add(tag='Losses/Policy Loss Predator',
                                      simple_value=float(np.mean(p_l)))  # policy_loss
                    summary.value.add(tag='Losses/Entropy Predator', simple_value=float(np.mean(e_l)))  # entropy
                    summary.value.add(tag='Losses/Grad Norm Predator', simple_value=float(np.mean(g_n)))  # grad_norms
                    summary.value.add(tag='Losses/Var Norm Predator', simple_value=float(np.mean(v_n)))  # var_norms

                    summary.value.add(tag='Perf/Reward Prey', simple_value=float(mean_reward_prey))  # avg reward
                    summary.value.add(tag='Perf/Value Prey',
                                      simple_value=float(mean_value_prey))  # avg episode value_predator
                    summary.value.add(tag='Losses/Value Loss Prey', simple_value=float(np.mean(v_l_prey)))  # value_loss
                    summary.value.add(tag='Losses/Policy Loss Prey',
                                      simple_value=float(np.mean(p_l_prey)))  # policy_loss
                    summary.value.add(tag='Losses/Entropy Prey', simple_value=float(np.mean(e_l_prey)))  # entropy
                    summary.value.add(tag='Losses/Grad Norm Prey', simple_value=float(np.mean(g_n_prey)))  # grad_norms
                    summary.value.add(tag='Losses/Var Norm Prey', simple_value=float(np.mean(v_n_prey)))  # var_norms

                    summary.value.add(tag='Strategy/Prey',
                                      simple_value=float(action_distribution_prey[0][0]))  # var_norms
                    # summary.value.add(tag='Strategy/Prey', simple_value=float(action_distribution_prey[0][1]))  # var_norms

                    summary.value.add(tag='Strategy/Predator', simple_value=float(action_distribution_predator[0][0]))
                    # summary.value.add(tag='Strategy/Predator', simple_value=float(action_distribution_predator[0][1]))

                    self.summary_writer.add_summary(summary, episode_count)

                    # self.summary_writer.add_graph(sess.graph, global_step=episode_count)
                    self.summary_writer.flush()
                    # exit()

            # Update episode count
            if self.is_chief:
                glbl_epis = sess.run(self.increment)
                # glbl_epis = sess.run(self.global_episodes)
                if glbl_epis % 500 == 0:
                    print("Global step @", glbl_epis, "*", batch_size)

                if glbl_epis % (500 / batch_size) == 0:
                    print("Copying on-line to target network")
                    sess.run([self.local_AC_predator.assign_op, self.local_AC_prey.assign_op])
            episode_count += 1

            # if total_steps > 2500:
            #    coord.request_stop()
