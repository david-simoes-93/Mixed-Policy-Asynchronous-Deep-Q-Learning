# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
#   tensorboard --logdir=worker_0:'./train_0'
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7',worker_8:'./train_8',worker_9:'./train_9',worker_10:'./train_10',worker_11:'./train_11'


import argparse
import os

import tensorflow as tf

from asyncdqn.QNetworkPolicy import QNetworkPolicy
from asyncdqn.QNetworkPolicySlow import QNetworkPolicySlow
from asyncdqn.QNetwork1step import QNetwork1Step
from asyncdqn.Slave import Worker
from asyncdqn.SlaveRPS import WorkerRPS
from asyncdqn.SlaveRandomRPS import WorkerRandomRPS
from simulator.GymPursuit import GymPursuit
from simulator.GymRPS import GymRPS
from simulator.GymRandomRPS import GymRandomRPS

max_episode_length = 2000

regular_pursuit = False
random_rps = False
pursuit_oscil = False
gt = True

# pursuit
if regular_pursuit:
    gamma = .99             # discount rate for advantage estimation and reward discounting
    width = 15              # 84
    height = 15
    a_size = 5              # Agent can move Left, Right, up down, or nothing
    number_of_cell_types = 3
    learning_rate = 1e-4    # this was 1e-5 as of 02/05/2017
# random rps
if random_rps:
    gamma = .9              # discount rate for advantage estimation and reward discounting
    width = 5
    height = 5
    a_size = 4              # Agent can move Left, Right, up down
    number_of_cell_types = 1
    learning_rate = 1e-5
# pursuit balance test
if pursuit_oscil:
    gamma = .99             # discount rate for advantage estimation and reward discounting
    width = 7               # 84
    height = 7
    a_size = 5              # Agent can move Left, Right, up down, or nothing
    number_of_cell_types = 3
    learning_rate = 1e-4
if gt:
    gamma = .9  # discount rate for advantage estimation and reward discounting
    width = 1
    height = 1
    a_size = 2  # Agent can move Left, Right, up down
    number_of_cell_types = 1
    learning_rate = 1e-5
    game = "tricky"

s_size = width * height     # Observations are greyscale frames of 84 * 84 * 1

load_model = False
model_path = './model_dist'
debug = False
use_lstm = False
use_conv_layers = False
save_gifs = False

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
)
parser.add_argument(
    "--slaves_per_url",
    type=str,
    default="1",
    help="Comma-separated list of maximum tasks within the job"
)
parser.add_argument(
    "--urls",
    type=str,
    default="localhost",
    help="Comma-separated list of hostnames"
)
parser.add_argument(
    "--num_predators",
    type=int,
    default=1,
    help="Set number of predators"
)
parser.add_argument(
    "--num_prey",
    type=int,
    default=1,
    help="Set number of prey"
)
parser.add_argument(
    "--alg",
    type=int,
    default=4,
    help="Which algorithm it is"
)
FLAGS, unparsed = parser.parse_known_args()

number_of_predators = FLAGS.num_predators
number_of_prey = FLAGS.num_prey

# Create a cluster from the parameter server and worker hosts.
hosts = []
for (url, max_per_url) in zip(FLAGS.urls.split(","), FLAGS.slaves_per_url.split(",")):
    for i in range(int(max_per_url)):
        hosts.append(url+":" + str(2210 + i))
cluster = tf.train.ClusterSpec({"a3c": hosts})
server = tf.train.Server(cluster, job_name="a3c", task_index=FLAGS.task_index)

tf.reset_default_graph()

# Create a directory to save models and episode playback gifs
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device(tf.train.replica_device_setter(worker_device="/job:a3c/task:%d" % FLAGS.task_index, cluster=cluster)):
    global_episodes = tf.contrib.framework.get_or_create_global_step()
    trainer_predator = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainer_prey = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if FLAGS.alg in [3, 4, 5]:
        master_network_predator = QNetworkPolicy(s_size, a_size, 'global_predator', None, use_conv_layers, use_lstm,
                                                 width, height, number_of_cell_types)  # Generate global network
        master_network_prey = QNetworkPolicy(s_size, a_size, 'global_prey', None, use_conv_layers, use_lstm,
                                             width, height, number_of_cell_types)  # Generate global network
    elif FLAGS.alg in [1,2]:
        master_network_predator = QNetworkPolicySlow(s_size, a_size, 'global_predator', None, use_conv_layers, use_lstm,
                                                 width, height, number_of_cell_types)  # Generate global network
        master_network_prey = QNetworkPolicySlow(s_size, a_size, 'global_prey', None, use_conv_layers, use_lstm,
                                             width, height, number_of_cell_types)  # Generate global network
    else:
        master_network_predator = QNetwork1Step(s_size, a_size, 'global_predator', None, use_conv_layers, use_lstm,
                                                     width, height, number_of_cell_types)  # Generate global network
        master_network_prey = QNetwork1Step(s_size, a_size, 'global_prey', None, use_conv_layers, use_lstm,
                                                 width, height, number_of_cell_types)  # Generate global network

    # Master declares worker for all slaves
    for i in range(len(hosts)):
        print("Initializing variables for slave ", i)
        if random_rps:
            if i == FLAGS.task_index:
                worker = WorkerRandomRPS(GymRandomRPS(2, width, width, debug), #GymRPS(),
                                i, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                                width, height, number_of_cell_types, use_lstm, use_conv_layers, save_gifs,
                                number_of_predators, number_of_prey, FLAGS.alg)
            else:
                WorkerRandomRPS(GymRandomRPS(2, width, width, debug), #GymRPS(),
                       i, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                       width, height, number_of_cell_types, use_lstm, use_conv_layers, save_gifs, number_of_predators,
                       number_of_prey, FLAGS.alg)
        if regular_pursuit or pursuit_oscil:
            if i == FLAGS.task_index:
                worker = Worker(GymPursuit(number_of_predators, number_of_prey, width, width, debug),
                                i, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                                width, height, number_of_cell_types, use_lstm, use_conv_layers, save_gifs, number_of_predators, number_of_prey)
            else:
                Worker(GymPursuit(number_of_predators, number_of_prey, width, width, debug),
                       i, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                       width, height, number_of_cell_types, use_lstm, use_conv_layers, save_gifs, number_of_predators, number_of_prey)
        if gt:
            if i == FLAGS.task_index:
                worker = WorkerRPS(GymRPS(game=game),
                                i, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                                width, height, number_of_cell_types, use_lstm, use_conv_layers, save_gifs,
                                number_of_predators, number_of_prey, FLAGS.alg)
            else:
                WorkerRPS(GymRPS(game=game),
                       i, s_size, a_size, trainer_predator, trainer_prey, model_path, global_episodes,
                       width, height, number_of_cell_types, use_lstm, use_conv_layers, save_gifs, number_of_predators,
                       number_of_prey, FLAGS.alg)

print("Starting session", server.target, FLAGS.task_index)
hooks=[tf.train.StopAtStepHook(last_step=100000)]
with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                       config=tf.ConfigProto(),  # config=tf.ConfigProto(log_device_placement=True),
                                       save_summaries_steps=5, save_summaries_secs=None,
                                       save_checkpoint_secs=600, checkpoint_dir=model_path, hooks=hooks) as mon_sess:
    print("Started session")
    try:
        worker.work(max_episode_length, gamma, mon_sess)
    except RuntimeError:
        print("Puff")

print("Done")
with open("policy"+str(FLAGS.task_index)+".txt", "a") as myfile:
    myfile.write(str(FLAGS)+" "+str(worker.policy_evolution)+'\n')
