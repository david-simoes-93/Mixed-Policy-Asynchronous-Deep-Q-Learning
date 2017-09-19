#!/usr/bin/python3

import random
import numpy as np
from simulator.Simulator2 import PursuitSim, PREDATOR


class GymRPS(object):
    def __init__(self, game="matching_pennies"):
        if game=="matching_pennies":
            self.numberOfActions = 2
            self.moves = ["HEADS", "TAILS"]
            self.R1 = [[1, -1],
                       [-1, 1]]
            self.R2 = [[-1, 1],
                       [1, -1]]
        elif game=="biased":
            self.numberOfActions = 2
            self.moves = ["Action 0", "Action 1"]
            self.R1 = [[1.00, 1.75],
                       [1.25, 1.00]]
            self.R2 = [[1.75, 1.00],
                       [1.00, 1.25]]
        elif game=="tricky":
            self.numberOfActions = 2
            self.moves = ["Action 0", "Action 1"]
            self.R1 = [[0, 3],
                       [1, 2]]
            self.R2 = [[3, 2],
                       [0, 1]]
        elif game=="rps":
            self.numberOfActions = 3
            self.moves = ["ROCK", "PAPER", "SCISSORS"]
            self.R1 = [[0, -1, 1],
                  [1, 0, -1],
                  [-1, 1, 0]]
            self.R2 = [[0, 1, -1],
                  [-1, 0, 1],
                  [1, -1, 0]]
        elif game=="nrps":
            self.numberOfActions = 4
            self.moves = ["NULL", "ROCK", "PAPER", "SCISSORS"]
            self.R1 = [[-1, -1, -1, -1],
                       [-1, 2, 1, 3],
                       [-1, 3, 2, 1],
                       [-1, 1, 3, 2]]
            self.R2 = [[-1, -1, -1, -1],
                       [-1, 2, 3, 1],
                       [-1, 1, 2, 3],
                       [-1, 3, 1, 2]]
        else:
            print("Unknown game: ", game)
            exit()

        self.debug = False

        self.number_of_predators = 1  # number of agents
        self.number_of_prey = 1
        self.input_size = 1

        self._screen = np.zeros([1, self.input_size])
        self._screen_prey = np.zeros([1, self.input_size])

        self.reward = 0  # current processed reward
        self.reward_prey = 0
        self.terminal_prey = False
        self.terminal = False  # whether simulation is on-going
        self.display = False

        self.actions = np.zeros(1)
        self.actions_prey = np.zeros(1)

    def new_episode(self):
        return self._screen, self._screen_prey, 0, 0, [0] * self.number_of_predators, [0] * self.number_of_prey, False

    def get_state(self):
        return self._screen, self._screen_prey

    def is_episode_finished(self):
        return self.terminal

    @property
    def screen(self):
        return self._screen

    @property
    def action_size(self):
        return self.numberOfActions

    @property
    def lives(self):
        return 1

    @property
    def state(self):
        return [0], [0], self.reward, self.reward_prey

    def birdseye_observe(self):
        return [0]

    def predator_observe(self):
        self.reward = self.R1[self.actions[0]][self.actions_prey[0]]

        return self._screen, self.reward, self.terminal

    def predator_act(self, actions):
        self.actions = actions
        self.terminal = True

    def observe(self):
        self.prey_observe()
        self.predator_observe()
        return self.state

    def prey_observe(self):
        self.reward_prey = self.R2[self.actions[0]][self.actions_prey[0]]

        return self._screen_prey, self.reward_prey, self.terminal_prey, [self.terminal]*self.number_of_prey

    def act(self, actions):
        self.actions = [actions[0]]
        self.actions_prey = [actions[1]]
        self.terminal = True
        self.terminal_prey = True

    def prey_act(self, actions):
        self.actions_prey = actions
        self.terminal_prey = True

    def close(self):
        pass
