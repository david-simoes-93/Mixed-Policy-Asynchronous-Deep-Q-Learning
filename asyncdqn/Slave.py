from time import time
import tensorflow as tf
from asyncdqn.QNetwork1step import QNetwork1Step
from asyncdqn.Helper import update_target_graph, discount, make_gif, get_empty_loss_arrays
import numpy as np
from time import sleep
import math


# Worker class
class Worker:
    def __init__(self, game, name, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                 width=15, height=15, number_of_cell_types=3, use_lstm=False, use_conv_layers=False, save_gifs = False,
                 number_of_predators=5, number_of_prey=5):
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
        self.local_AC_predator = QNetwork1Step(s_size, a_size, self.name + "_predator", trainer_predator, use_conv_layers, use_lstm,
                                            width, height, number_of_cell_types)
        self.local_AC_prey = QNetwork1Step(s_size, a_size, self.name + "_prey", trainer_prey, use_conv_layers, use_lstm, width,
                                        height, number_of_cell_types)
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

        self.prey_doing_auto_action = True
        self.prey_learning = True
        self.predator_learning = True

        self.exploration_rate_predators = 1.0
        self.exploration_rate_prey = 0.0

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
        #print(self.number, "Train")
        """rollout.append([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                       1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0.], 3, 1, None, True, 0])"""

        rollout = np.array(rollout)
        observations = rollout[:, 0]                    # state t
        actions = rollout[:, 1]                         # action taken at timestep t
        rewards = rollout[:, 2]                         # reward t
        #next_observations = np.vstack(rollout[:, 3])   # state t+1
        terminals = rollout[:, 4]                       # whether timestep t was terminal
        next_max_q = rollout[:, 5]                      # the best Q value of state t+1 as calculated by target network

        #if terminals[-1]:
        #    print(actions[-1], rewards[-1])

        """if self.use_lstm:
            ""for pred_index in range(self.number_of_predators):
                max_q_t_plus_1 = np.max(sess.run(self.local_AC_predator.target.value,
                                     feed_dict={
                                         self.local_AC_predator.target.inputs: [previous_screen_predator[pred_index]],
                                         self.local_AC_predator.target.state_in[0]: rnn_state_predator[pred_index],
                                         self.local_AC_predator.target.state_in[1]: rnn_state_predator[pred_index]})[
                                0, 0])""
            pass
        else:
            max_q_t_plus_1 = sess.run(ac_network.target_network.best_q,
                          feed_dict={ac_network.target_network.inputs: next_observations})

        """

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
                         ac_network.state_in[1]: rnn_state[1]}
        else:
            feed_dict = {ac_network.target_q_t: discounted_rewards,
                         #ac_network.computed_q_t: values,
                         ac_network.network.inputs: np.vstack(observations),
                         ac_network.actions: actions}
        v_l, q_value, q_acted, g_n, v_n, _ = sess.run([ac_network.loss, ac_network.network.value, ac_network.q_acted,
                                               ac_network.grad_norms,
                                               ac_network.var_norms,
                                               ac_network.apply_grads],
                                              feed_dict=feed_dict)
        #v_l, q_value, q_acted, g_n, v_n = 0,0,0,0,0

        """index = len(observations) - 1
        self.print_screen_formatted(observations[index])
        print(actions[index], rewards[index], terminals[index])
        #self.print_screen_formatted(next_observations[index])
        print(next_max_q[index], discounted_rewards[index])
        print(v_l, q_value[index], q_acted[index])
        #input("continue?")"""

        return v_l / len(rollout), 0, 0, g_n, v_n

    def process(self, screen):
        one_out_of_n = []
        for s in screen:
            t = np.zeros(self.s_size * self.number_of_cell_types)
            for i in range(len(s)):
                cell = int(s[i] * self.number_of_cell_types + 0.01)
                if cell != 0:
                    t[i * self.number_of_cell_types + cell - 1] = 1
            one_out_of_n.append(t)
        return one_out_of_n

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
        episode_count = 0 #sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        # with sess.as_default(), sess.graph.as_default():
        prev_clock = time()
        if coord is None:
            coord = sess

        sess.run(self.local_AC_predator.assign_op)
        sess.run(self.local_AC_prey.assign_op)
        print(self.number, "Resetting target network")

        #prev_vals = sess.run(self.local_AC_predator.local_vars)
        while not coord.should_stop():
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
            action_indexes = list(range(self.env.numberOfActions))

            self.env.new_episode()
            previous_screen_predator, previous_screen_prey = self.env.get_state()  # .screen_buffer
            episode_frames.append(self.env.birdseye_observe())
            if not self.use_conv_layers:
                previous_screen_predator = self.process(previous_screen_predator)
                previous_screen_prey = self.process(previous_screen_prey)
            both_terminal = 0

            # Initial state; initialize values for the LSTM network
            action_predator, action_distribution_predator, value_predator, rnn_state_predator = \
                self.take_action_from_network(sess, self.local_AC_predator.network, self.number_of_predators,
                                              previous_screen_predator, action_indexes,
                                              [], [], [])
            action_prey, action_distribution_prey, value_prey, rnn_state_prey = \
                self.take_action_from_network(sess, self.local_AC_prey.network, self.number_of_prey,
                                              previous_screen_prey, action_indexes,
                                              [], [], [])
            prev_terminal_each = [False]*self.number_of_prey

            # Go a turn without moving
            action_predator = [0] * self.number_of_predators
            action_prey = [0] * self.number_of_prey
            episode_buffer_prey_counter = 0
            #times_measurement = []
            v_l_prey, p_l_prey, e_l_prey, g_n_prey, v_n_prey = get_empty_loss_arrays(self.number_of_prey)
            v_l, p_l, e_l, g_n, v_n = get_empty_loss_arrays(self.number_of_predators)
            while episode_step_count < max_episode_length:
                #sleep(1)
                end_epsilon_probs = [0.01, 0.1, 0.5]
                self.exploration_rate_predators -= (1.0-end_epsilon_probs[self.number])/5000000.0
                if self.exploration_rate_predators < end_epsilon_probs[self.number]:
                    self.exploration_rate_predators = end_epsilon_probs[self.number]

                ############# Predator Turn #############

                # Watch environment
                current_screen_predator, reward, terminal_predator = self.env.predator_observe()
                reward = np.sum(reward)
                episode_reward_predator += reward
                episode_frames.append(self.env.birdseye_observe())
                if not self.use_conv_layers:
                    current_screen_predator = self.process(current_screen_predator)

                # get target network values
                next_max_q = sess.run(self.local_AC_predator.target_network.best_q,
                         feed_dict={self.local_AC_predator.target_network.inputs: current_screen_predator})

                # Store environment
                for pred_index in range(self.number_of_predators):
                    episode_buffer_predator[pred_index].append([previous_screen_predator[pred_index],
                                                                action_predator[pred_index],
                                                                reward,
                                                                current_screen_predator[pred_index],
                                                                terminal_predator,
                                                                next_max_q[pred_index]])
                    episode_values_predator[pred_index].append(np.max(value_predator[pred_index]))
                previous_screen_predator = current_screen_predator

                # If prey just suicided last turn and we are terminal, then lets waste a turn
                if terminal_predator:
                    both_terminal += 1
                    action_predator = [0] * self.number_of_predators
                # random exploration
                elif np.random.random() < self.exploration_rate_predators:
                    action_predator = np.random.choice(action_indexes, self.number_of_predators)
                # Otherwise, lets judge and execute a move
                else:
                    action_predator, action_distribution_predator, value_predator, rnn_state_predator = \
                        self.take_action_from_network(sess, self.local_AC_predator.network, self.number_of_predators,
                                                      previous_screen_predator, action_indexes,
                                                      action_distribution_predator, value_predator, rnn_state_predator)
                self.env.predator_act(action_predator)

                # If the episode hasn't ended, but the experience buffer is full, then we make an update step using that experience rollout.
                if self.predator_learning and len(episode_buffer_predator[0]) == 30 and episode_step_count < max_episode_length - 1:
                    for pred_index in range(self.number_of_predators):
                        v_l[pred_index], p_l[pred_index], e_l[pred_index], g_n[pred_index], v_n[pred_index] = \
                            self.train(episode_buffer_predator[pred_index], sess, gamma, self.local_AC_predator)

                    episode_buffer_predator = []
                    for _ in range(self.number_of_predators):
                        episode_buffer_predator.append([])

                    sess.run(self.update_local_ops_predator)

                # Measure time and increase episode step count
                total_steps += 1
                if total_steps % 2000 == 0:
                    new_clock = time()
                    print(2000.0 / (new_clock - prev_clock), "it/s,   ")
                    prev_clock = new_clock
                    #times_measurement = []
                episode_step_count += 1

                # Copy online -> target networks
                if sess.run(self.increment) % 5000 == 0:
                    sess.run([self.local_AC_predator.assign_op, self.local_AC_prey.assign_op])
                    """print(self.number, "Updating target network")
                    debug_state = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
                                   0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                    debug_q = sess.run(self.local_AC_predator.target_network.value,
                                       feed_dict={self.local_AC_predator.target_network.inputs: [debug_state]})
                    print(self.number, debug_q)"""

                # If both prey and predator have ackowledged game is over, then break from episode
                if both_terminal == 2:
                    break

                ############# Prey Turn #############
                if True:
                    # Watch environment
                    current_screen_prey_raw, reward, terminal, terminal_each = self.env.prey_observe()
                    reward = [int(not x)-1 for x in terminal_each]      # make prey receive independent rewards!
                    #reward = np.sum(reward)
                    episode_reward_prey += np.sum(reward)
                    episode_frames.append(self.env.birdseye_observe())

                    if self.prey_doing_auto_action:
                        if not self.use_conv_layers:
                            current_screen_prey = self.process(current_screen_prey_raw)
                        else:
                            current_screen_prey = current_screen_prey_raw

                        # get target network values
                        next_max_q_prey = sess.run(self.local_AC_prey.target_network.best_q,
                                              feed_dict={self.local_AC_prey.target_network.inputs: current_screen_prey})

                        # Store environment
                        episode_buffer_prey_counter += 1
                        for prey_index in range(self.number_of_prey):
                            if not prev_terminal_each[prey_index]:  # if prey used to be alive
                                episode_buffer_prey[prey_index].append([previous_screen_prey[prey_index],
                                                                        action_prey[prey_index],
                                                                        reward[prey_index],
                                                                        current_screen_prey[prey_index],
                                                                        terminal_each[prey_index],
                                                                        next_max_q_prey[prey_index]])
                                episode_values_prey[prey_index].append(np.max(value_prey[prey_index]))
                        previous_screen_prey = current_screen_prey
                        prev_terminal_each = terminal_each

                    # If predator just caught everyone last turn and we are terminal, then lets waste a turn
                    if terminal:
                        both_terminal += 1
                        action_prey = [0] * self.number_of_prey
                    # random exploration
                    elif np.random.random() < self.exploration_rate_prey:
                        action_prey = np.random.choice(action_indexes, self.number_of_prey)
                    elif not self.prey_doing_auto_action:
                        action_prey = self.env.smart_prey_move(current_screen_prey_raw)
                    # Otherwise, lets judge and execute a move
                    else:
                        action_prey, action_distribution_prey, value_prey, rnn_state_prey = \
                            self.take_action_from_network(sess, self.local_AC_prey.network, self.number_of_prey,
                                                          previous_screen_prey, action_indexes,
                                                          action_distribution_prey, value_prey, rnn_state_prey)
                    self.env.prey_act(action_prey)

                    # If the episode hasn't ended, but the experience buffer is full, then we make an update step using that experience rollout.
                    if self.prey_doing_auto_action and self.prey_learning and episode_buffer_prey_counter == 30 and episode_step_count < max_episode_length - 1:
                        for prey_index in range(self.number_of_prey):
                            # If this prey was alive since the last update step (meaning it has recorded episodes in its buffer), we update
                            if len(episode_buffer_prey[prey_index]) != 0:
                                v_l_prey[prey_index], p_l_prey[prey_index], e_l_prey[prey_index], g_n_prey[prey_index], v_n_prey[prey_index] = \
                                    self.train(episode_buffer_prey[prey_index], sess, gamma, self.local_AC_prey)

                        episode_buffer_prey = []
                        episode_buffer_prey_counter = 0
                        for _ in range(self.number_of_prey):
                            episode_buffer_prey.append([])

                        sess.run(self.update_local_ops_prey)

                    # Measure time and increase episode step count
                    total_steps += 1
                    if total_steps % 2000 == 0:
                        new_clock = time()
                        #self.speed_up = np.mean(times_measurement)
                        print(2000.0 / (new_clock - prev_clock), "it/s,   ") #, self.speed_up,"speed-up")
                        prev_clock = new_clock
                        #times_measurement=[]
                    episode_step_count += 1

                    # If both prey and predator have ackowledged game is over, then break from episode
                    if both_terminal == 2:
                        break

            # print("0ver ",episode_count)
            self.episode_rewards_predator.append(episode_reward_predator)
            self.episode_rewards_prey.append(episode_reward_prey)
            self.episode_lengths.append(episode_step_count)
            self.episode_mean_values_predator.append(np.mean(episode_values_predator))
            if self.prey_doing_auto_action:
                this_episode_values = []
                for each_prey_value in episode_values_prey:
                    if len(each_prey_value)!=0:
                        this_episode_values.append(np.mean(each_prey_value))
                self.episode_mean_values_prey.append(np.mean(this_episode_values))
            else:
                self.episode_mean_values_prey.append(0)

            # Update the network using the experience buffer at the end of the episode.
            if self.predator_learning:
                if len(episode_buffer_predator[0]) != 0:
                    for pred_index in range(self.number_of_predators):
                        v_l[pred_index], p_l[pred_index], e_l[pred_index], g_n[pred_index], v_n[pred_index] = \
                            self.train(episode_buffer_predator[pred_index], sess, gamma, self.local_AC_predator)

            # Update the network using the experience buffer at the end of the episode.
            if self.prey_learning:
                for prey_index in range(self.number_of_prey):
                    if len(episode_buffer_prey[prey_index]) != 0:
                        #print(len(episode_buffer_prey[prey_index]), episode_buffer_prey[prey_index])
                        v_l_prey[prey_index], p_l_prey[prey_index], e_l_prey[prey_index], g_n_prey[prey_index], v_n_prey[prey_index] = \
                            self.train(episode_buffer_prey[prey_index], sess, gamma, self.local_AC_prey)



            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if episode_count % 5 == 0 and episode_count != 0:
                # Save .GIF of episode
                if self.is_chief and self.save_gifs and episode_count % 100 == 0:
                    make_gif(episode_frames, './frames/image' + str(episode_count) + '.gif', 0.2, self.width, self.height)
                    print("Saved .GIF")

                # Save current model
                if self.is_chief and saver is not None and episode_count % 250 == 0:
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")

                # Save statistics for TensorBoard
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_reward_predator = np.mean(self.episode_rewards_predator[-5:])
                mean_value_predator = np.mean(self.episode_mean_values_predator[-5:])
                mean_reward_prey = np.mean(self.episode_rewards_prey[-5:])
                mean_value_prey = np.mean(self.episode_mean_values_prey[-5:])

                summary = tf.Summary()

                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))                       # avg episode length

                summary.value.add(tag='Perf/Reward Predator', simple_value=float(mean_reward_predator))     # avg reward
                summary.value.add(tag='Perf/Value Predator', simple_value=float(mean_value_predator))       # avg episode value_predator
                summary.value.add(tag='Losses/Value Loss Predator', simple_value=float(np.mean(v_l)))       # value_loss
                summary.value.add(tag='Losses/Policy Loss Predator', simple_value=float(np.mean(p_l)))      # policy_loss
                summary.value.add(tag='Losses/Entropy Predator', simple_value=float(np.mean(e_l)))          # entropy
                summary.value.add(tag='Losses/Grad Norm Predator', simple_value=float(np.mean(g_n)))        # grad_norms
                summary.value.add(tag='Losses/Var Norm Predator', simple_value=float(np.mean(v_n)))         # var_norms

                summary.value.add(tag='Perf/Reward Prey', simple_value=float(mean_reward_prey))             # avg reward
                summary.value.add(tag='Perf/Value Prey', simple_value=float(mean_value_prey))               # avg episode value_predator
                summary.value.add(tag='Losses/Value Loss Prey', simple_value=float(np.mean(v_l_prey)))      # value_loss
                summary.value.add(tag='Losses/Policy Loss Prey', simple_value=float(np.mean(p_l_prey)))     # policy_loss
                summary.value.add(tag='Losses/Entropy Prey', simple_value=float(np.mean(e_l_prey)))         # entropy
                summary.value.add(tag='Losses/Grad Norm Prey', simple_value=float(np.mean(g_n_prey)))       # grad_norms
                summary.value.add(tag='Losses/Var Norm Prey', simple_value=float(np.mean(v_n_prey)))        # var_norms
                self.summary_writer.add_summary(summary, episode_count)

                self.summary_writer.flush()

            # Update episode count
            if self.is_chief:
                #sess.run(self.increment)
                print("Global step @", sess.run(self.global_episodes), " epsilon =", self.exploration_rate_predators)
            episode_count += 1

            #if total_steps > 2500:
            #    coord.request_stop()
