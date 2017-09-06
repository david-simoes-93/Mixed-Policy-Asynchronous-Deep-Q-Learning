import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from asyncdqn.Helper import normalized_columns_initializer


# 1-step Q-network
class QNetworkPolicy:
    def make_network(self, s_size, a_size, use_conv_layers, use_lstm, width, height, number_of_cell_types, trainable=True):
        network = type('', (), {})()

        # Input and visual encoding layers
        if use_conv_layers:
            network.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            network.imageIn = tf.reshape(network.inputs, shape=[-1, width, height, 1])
            network.conv = slim.conv2d(activation_fn=tf.nn.elu,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    # normalized_columns_initializer(0.01),
                                    inputs=network.imageIn, num_outputs=8,
                                    kernel_size=[3, 3], stride=[1, 1], padding='VALID', trainable=trainable)
            network.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                     # normalized_columns_initializer(0.01),
                                     inputs=network.imageIn, num_outputs=4,
                                     kernel_size=[1, 1], stride=[1, 1], padding='VALID', trainable=trainable)
            hidden = slim.fully_connected(slim.flatten(network.conv2), 150,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation_fn=tf.nn.elu, trainable=trainable)
            hidden2 = slim.fully_connected(hidden, 150, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.nn.elu, trainable=trainable)

        else:
            network.inputs = tf.placeholder(shape=[None, s_size * number_of_cell_types], dtype=tf.float32)
            hidden = slim.fully_connected(network.inputs, 150,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation_fn=tf.nn.elu, trainable=trainable)
            hidden2 = slim.fully_connected(hidden, 150, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.nn.elu, trainable=trainable)

            #todo
            #hidden2 = slim.fully_connected(network.inputs, 9, weights_initializer=tf.contrib.layers.xavier_initializer(),
            #                               activation_fn=tf.nn.elu, trainable=trainable)

        if use_lstm:
            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            network.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            network.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden2, [0])  # converts hidden layer [256] to [1, 256]
            step_size = tf.shape(network.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in,
                                                         sequence_length=step_size, time_major=False, trainable=trainable)
            lstm_c, lstm_h = lstm_state
            network.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            network.value = slim.fully_connected(rnn_out, a_size, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=None, trainable=trainable)
            network.policy_before_softmax = slim.fully_connected(rnn_out, a_size, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=None, trainable=trainable)
        else:
            network.value = slim.fully_connected(hidden2, a_size, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=None, trainable=trainable)
            network.policy_before_softmax = slim.fully_connected(hidden2, a_size, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=None, trainable=trainable)
        network.policy = tf.nn.softmax(network.policy_before_softmax)
        network.policy_slow = network.policy # ignore

        network.best_q = tf.reduce_max(network.value,1)
        network.best_action_index = tf.arg_max(network.value,1)

        return network

    def __init__(self, s_size, a_size, scope, trainer, use_conv_layers=False, use_lstm=False, width=15, height=15, number_of_cell_types=3):
        with tf.variable_scope(scope):
            print("Scope", scope)
            with tf.variable_scope("regular"):
                self.network = self.make_network(s_size, a_size, use_conv_layers, use_lstm, width, height, number_of_cell_types)
            with tf.variable_scope("target"):
                self.target_network = self.make_network(s_size, a_size, use_conv_layers, use_lstm, width, height,
                                                        number_of_cell_types, trainable=True)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global_predator' and scope != 'global_prey':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int64)                 # Index of actions taken
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)    # 1-hot tensor of actions taken

                # losses!
                #self.computed_q_t = tf.placeholder('float32', [None], name='target_q_t')
                self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
                self.q_acted = tf.reduce_sum(self.network.value * self.actions_onehot, reduction_indices=1, name='q_acted')

                #ignore this
                self.target_policy_slow_t = tf.placeholder('float32', [None, a_size], name='target_policy_slow_t')

                self.target_policy_t = tf.placeholder('float32', [None, a_size], name='target_policy_t')
                self.loss_policy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.network.policy_before_softmax,
                                                                       labels=self.target_policy_t))

                """self.condition = tf.equal(self.actions, self.network.best_action_index)
                anti_value = (1 / (a_size - 1))
                self.anti_vector = tf.constant(np.full(a_size, 1 / anti_value), dtype=tf.float32)
                self.anti_onehot = tf.scalar_mul(anti_value, self.actions_onehot)
                self.topkek = self.anti_vector - self.anti_onehot
                self.vector = tf.where(self.condition, self.actions_onehot, self.anti_vector - self.anti_onehot)
                self.eta = tf.where(self.condition, tf.constant(np.full(1, 1.0/1000.0), dtype=tf.float32),
                                    tf.constant(np.full(1, 1.0/2000.0), dtype=tf.float32)) # todo: this shouldnt be dim 1, should be 0, so that
                ## todo we can eventually do minibatches

                #batch_size = 1 #30
                #self.eta = tf.where(self.condition, tf.constant(1e-4, shape=[batch_size, 2], dtype=tf.float32),
                #                    tf.constant((1e-4) / 2, shape=[batch_size, 2], dtype=tf.float32))
                # todo: this shouldnt be dim 30, should be 'batch_size'

                #self.target_policy = tf.multiply((1-self.eta), self.network.policy)
                #self.target_policy = self.eta * self.vector
                self.target_policy = (1 - self.eta) * self.network.policy + self.eta * self.vector"""

                self.loss = tf.reduce_mean(tf.square(self.target_q_t - self.q_acted) + self.loss_policy, name='loss')
                #                           tf.square(self.target_policy_t - self.network.policy), name='loss')
                """self.test1 = tf.reduce_sum(tf.square(self.target_policy_t - self.network.policy),
                                                                   reduction_indices=1)
                self.test2 = tf.square(self.target_policy_t - self.network.policy)
                self.test3 = tf.square(self.target_q_t - self.q_acted)"""
                #self.loss = tf.reduce_mean(tf.square(self.target_q_t - self.q_acted) +
                #                           tf.reduce_sum(tf.square(self.target_policy - self.network.policy),
                #                                                   reduction_indices=1), name='loss')

                #    prey [0] = 0.75,      prey[1]=0.25
                # predator[0] = 0.25   predator[1]=0.75
                # 0.25 0.75 -- 0.75 0.25

                # Get gradients from local network using local losses
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope))
                self.gradients = tf.gradients(self.loss, self.local_vars)
                self.var_norms = tf.global_norm(self.local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                if "predator" in scope:
                    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_predator/regular')
                    target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global_predator/target")
                elif "prey" in scope:
                    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_prey/regular')
                    target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global_prey/target")
                else:
                    print("Error on scope build", scope)
                    exit()

                # set global target network to be the same as local network
                weights = slim.get_trainable_variables(scope=scope + "/regular")
                self.assign_op = {}
                for w, t_w in zip(weights, target_weights):
                    #print(w, t_w)
                    self.assign_op[w.name] = t_w.assign(w)

                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
