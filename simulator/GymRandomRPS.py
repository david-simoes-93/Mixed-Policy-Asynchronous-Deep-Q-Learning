#!/usr/bin/python3

import random
import numpy as np
from simulator.RandomRPS import RandomRPSSim


class GymRandomRPS(object):
    def __init__(self, number_of_predators, screen_width, screen_height, gui):
        self.numberOfActions = 4
        # self.numberOfMapCells = 3           # agent, prey, wall

        self.debug = False

        self.number_of_agents = number_of_predators  # number of agents

        self.width = screen_width
        self.height = screen_height

        self.input_size = screen_width * screen_height  # *self.numberOfMapCells

        self._screen = np.zeros([number_of_predators, self.input_size])

        self.reward = [0]*number_of_predators  # current processed reward
        self.terminal = True  # whether simulation is on-going
        self.display = False

        self.pursuit_sim = RandomRPSSim(self.width, self.height, self.number_of_agents, gui=gui)
        self.moves = ["DOWN ", " UP  ", "LEFT ", "RIGHT"] #aka LOSE ROCK PAPER SCISSORS

    def new_episode(self):
        self.pursuit_sim.reset()

    def get_state(self):
        return [self._screen[0]], [self._screen[1]]

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
        return [self._screen[0]], [self._screen[1]], [self.reward[0]], [self.reward[1]]

    def birdseye_observe(self):

        return np.reshape(self.pursuit_sim.get_state_global(), [-1])

    def observe(self):
        self._screen = np.empty([self.number_of_agents, self.input_size])
        self.reward = [0]*self.number_of_agents
        for i in range(self.number_of_agents):
            screen, terml, self.reward[i] = self.pursuit_sim.get_state(i)
            self._screen[i] = np.reshape(screen, [-1])

        self.terminal = False
        return self.state

    def act(self, actions):
        for i in range(self.number_of_agents):
            self.pursuit_sim.move(i, actions[i])

        if self.pursuit_sim.playingRPS:
            #print("gg")
            self.pursuit_sim.reset()
            #exit()

    def close(self):
        pass
