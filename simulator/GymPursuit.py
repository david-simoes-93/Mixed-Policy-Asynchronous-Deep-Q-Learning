#!/usr/bin/python3

import random
import numpy as np
from simulator.Simulator2 import PursuitSim, PREDATOR


class GymPursuit(object):
    def __init__(self, number_of_predators, number_of_prey, screen_width, screen_height, gui):
        self.numberOfActions = 5
        # self.numberOfMapCells = 3           # agent, prey, wall

        self.debug = False

        self.number_of_predators = number_of_predators  # number of agents

        self.width = screen_width
        self.height = screen_height

        # self.jal = jal
        self.number_of_prey = number_of_prey
        # self.independent_rewards = independent_rewards

        self.input_size = screen_width * screen_height  # *self.numberOfMapCells

        self._screen = np.zeros([number_of_predators, self.input_size])
        self._screen_prey = np.zeros([number_of_prey, self.input_size])

        self.reward = 0  # current processed reward
        self.reward_prey = 0
        self.terminal = True  # whether simulation is on-going
        self.display = False

        self.pursuit_sim = PursuitSim(self.width, self.height, self.number_of_predators, self.number_of_prey, gui=gui)
        self.moves = ["  X  ", "DOWN ", " UP  ", "LEFT ", "RIGHT"]

        self.learning_test_values_mix = [0]
        self.learning_test_values_prey_mix = [0]

    def new_episode(self):
        self.pursuit_sim.reset()
        self.predator_observe()
        self.prey_observe()

        return self.screen, self._screen_prey, 0, 0, [0] * self.number_of_predators, [
            0] * self.number_of_prey, self.terminal

    def get_state(self):
        return self._screen, self._screen_prey

    def is_episode_finished(self):
        return self.terminal

    def smart_prey_move(self, s_t):
        actions = [0] * self.number_of_prey
        for i in range(self.number_of_prey):
            predators = []
            me = [int(self.height / 2), int(self.width / 2)]
            #print("me at",me)
            for y in range(self.height):
                for x in range(self.width):
                    index = x + y * self.width
                    #print(s_t[i][index], PREDATOR, s_t[i][index] == PREDATOR)
                    if s_t[i][index] == PREDATOR:
                        predators.append([x, y])
            #print("found pred at",predators[0])
            minDist = self.width * self.height
            #print("me", me)
            closestPredator = [me[0], me[1]]
            for b in predators:
                if abs(b[0] - me[0]) + abs(b[1] - me[1]) < minDist:
                    minDist = abs(b[0] - me[0]) + abs(b[1] - me[1])
                    closestPredator = [b[0], b[1]]
            relX = closestPredator[0] - me[0]
            relY = closestPredator[1] - me[1]
            if abs(relX) < abs(relY):
                actions[i] = 4 if relX < 0 else 3 if relX > 0 else (4 if random.random() < 0.5 else 3)
            elif abs(relX) > abs(relY):
                actions[i] = 1 if relY < 0 else 2 if relY > 0 else (1 if random.random() < 0.5 else 2)
            else:
                if relX > 0 and relY > 0:
                    actions[i] = 3 if random.random() < 0.5 else 2
                elif relX > 0 and relY < 0:
                    actions[i] = 3 if random.random() < 0.5 else 1
                elif relX < 0 and relY > 0:
                    actions[i] = 4 if random.random() < 0.5 else 2
                else:
                    actions[i] = 4 if random.random() < 0.5 else 1
        return actions

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
        return np.reshape(self.pursuit_sim.get_state_global(), [-1])

    def predator_observe(self):
        self._screen = np.empty([self.number_of_predators, self.input_size])
        for i in range(self.number_of_predators):
            screen, terml, rwrd = self.pursuit_sim.get_state_predator(i)
            self._screen[i] = np.reshape(screen, [-1])
        self.reward = rwrd
        self.terminal = terml
        return self.state

    def predator_act(self, actions):
        for i in range(self.number_of_predators):
            self.pursuit_sim.movePredator(i, actions[i])
        self.pursuit_sim.checkCollisionsPredator()
        self.pursuit_sim.checkPredatorCaughtPrey_predator()

    def prey_observe(self):
        self._screen_prey = np.empty([self.number_of_prey, self.input_size])
        terminal_each = [False]*self.number_of_prey
        for i in range(self.number_of_prey):
            screen, terminal_each[i], terminal, rwrd = self.pursuit_sim.get_state_prey(i)
            if terminal_each[i]:
                self._screen_prey[i] = [0] * (self.width * self.height)
                #print(self._screen_prey[i])
            else:
                self._screen_prey[i] = np.reshape(screen, [-1])
        self.terminal = terminal
        self.reward_prey = rwrd
        #print(self._screen_prey)
        return self._screen_prey, self.reward_prey, self.terminal, terminal_each

    def prey_act(self, actions):
        for i in range(self.number_of_prey):
            self.pursuit_sim.movePrey(i, actions[i])
        self.pursuit_sim.checkCollisionsPrey()
        self.pursuit_sim.checkPredatorCaughtPrey_prey()

    def close(self):
        pass
