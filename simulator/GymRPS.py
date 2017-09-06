#!/usr/bin/python3

import random
import numpy as np
from simulator.Simulator2 import PursuitSim, PREDATOR


class GymRPS(object):
    def __init__(self, number_of_predators=1, number_of_prey=1, numberOfActions=2):
        self.numberOfActions = numberOfActions
        self.moves = ["HEADS", "TAILS"]

        #self.numberOfActions = 3
        #self.moves = ["ROCK", "PAPER", "SCISSORS"]

        self.debug = False

        self.number_of_predators = number_of_predators  # number of agents
        self.number_of_prey = number_of_prey

        self.input_size = 1  # *self.numberOfMapCells

        self._screen = np.zeros([number_of_predators, self.input_size])
        self._screen_prey = np.zeros([number_of_prey, self.input_size])

        self.reward = 0  # current processed reward
        self.reward_prey = 0
        self.terminal_prey = False
        self.terminal = False  # whether simulation is on-going
        self.display = False

        self.R1=[[1.00, 1.75],
            [1.25, 1.00]] # biased
        self.R2=[[1.75, 1.00],
            [1.00, 1.25]]

        self.R1 = [[-1, -1, -1, -1],
                   [-1, 2, 1, 3],
                   [-1, 3, 2, 1],
                   [-1, 1, 3, 2]]  # rock paper scissors
        self.R2 = [[-1, -1, -1, -1],
                   [-1, 2, 3, 1],
                   [-1, 1, 2, 3],
                   [-1, 3, 1, 2]]  # rock paper scissors

        self.actions = np.zeros(number_of_predators)
        self.actions_prey = np.zeros(number_of_predators)

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
        return self._screen, self.reward, self.terminal

    def birdseye_observe(self):
        return [0]

    def predator_observe(self):
        if self.numberOfActions == 3:
            if (self.actions[0] == 0 and self.actions_prey[0] == 1) or \
               (self.actions[0] == 1 and self.actions_prey[0] == 2) or \
               (self.actions[0] == 2 and self.actions_prey[0] == 3):
                self.reward = -1
            elif (self.actions[0] == 0 and self.actions_prey[0] == 2) or \
                 (self.actions[0] == 1 and self.actions_prey[0] == 0) or \
                 (self.actions[0] == 2 and self.actions_prey[0] == 1):
                self.reward = 1
            else:
                self.reward = 0
        else:
            if self.actions[0] != self.actions_prey[0]:
                self.reward = 1
            else:
                self.reward = -1
            self.reward = self.R1[self.actions[0]][self.actions_prey[0]]

        return self._screen, self.reward, self.terminal

    def predator_act(self, actions):
        self.actions = actions
        self.terminal = True

    def prey_observe(self):
        if self.numberOfActions == 3:
            if (self.actions_prey[0] == 0 and self.actions[0] == 1) or \
               (self.actions_prey[0] == 1 and self.actions[0] == 2) or \
               (self.actions_prey[0] == 2 and self.actions[0] == 3):
                self.reward_prey = -1
            elif (self.actions_prey[0] == 0 and self.actions[0] == 2) or \
                 (self.actions_prey[0] == 1 and self.actions[0] == 0) or \
                 (self.actions_prey[0] == 2 and self.actions[0] == 1):
                self.reward_prey = 1
            else:
                self.reward_prey = 0
        else:
            if self.actions[0] == self.actions_prey[0]:
                self.reward_prey = 1
            else:
                self.reward_prey = -1
            self.reward_prey = self.R2[self.actions[0]][self.actions_prey[0]]

        return self._screen_prey, self.reward_prey, self.terminal_prey, [self.terminal]*self.number_of_prey

    def prey_act(self, actions):
        self.actions_prey = actions
        self.terminal_prey = True

    def close(self):
        pass
