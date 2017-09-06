import numpy as np
import pygame
from time import sleep

BLACK = np.array((0, 0, 0),dtype=np.uint8)           # obstacle
WHITE = np.array((255, 255, 255),dtype=np.uint8)     # clear
GREEN = np.array((0, 255, 0),dtype=np.uint8)         # prey
BLUE = np.array((0, 0, 255),dtype=np.uint8)          # predator
colors = [BLUE, WHITE]
WIDTH = 11
HEIGHT = 11
MARGIN = 1

AGENT = 1


class RandomRPSSim(object):
    def __init__(self, width, height, team, gui=False): # 10 1
        self.turn = 0

        self.imageCounter = 0
        self.saveImage = False

        self.width = width
        self.height = height

        self.map = np.zeros([width, height])
        self.team = [[0, 0] for _ in range(team)]  # np.zeros([predators, 2], dtype=int)

        self.rewards = [0] * team

        self.rps_actions = [0] * team

        self.terminal = False
        self.playingRPS = False

        self.gui = gui
        if gui:
            pygame.init()
            self.screen = pygame.display.set_mode([(WIDTH + MARGIN) * width, (HEIGHT + MARGIN) * height])
            pygame.display.set_caption("RandomRPS")
            #self.paint_map()
            # self.prev_debug_flag=0

        self.reset()

        self.R1 = [[-1, -1, -1, -1],
              [-1, 2, 1, 3],
              [-1, 3, 2, 1],
              [-1, 1, 3, 2]]  # rock paper scissors
        self.R2 = [[-1, -1, -1, -1],
              [-1, 2, 3, 1],
              [-1, 1, 2, 3],
              [-1, 3, 1, 2]]  # rock paper scissors

        #self.playingRPS = False

    def paint_map(self):
        #print("painting")
        if not self.gui:
            return

        self.screen.fill(BLACK)
        pred_count = 0
        for row in range(self.height):
            for column in range(self.width):
                color = colors[int(self.map[column][row] * (len(colors) - 1) + 0.01)]
                if int(self.map[column][row] * (len(colors) - 1) + 0.01) == 1:
                    pred_count += 1
                pygame.draw.rect(self.screen, color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH, HEIGHT])
        pygame.display.flip()

        if self.saveImage:
            pygame.image.save(pygame.display.get_surface(), "image" + str(self.imageCounter).zfill(4) + ".PNG")
            self.imageCounter += 1

    def reset(self):
        self.map = np.random.random([self.width, self.height])/2
        self.team = [[0, 0] for _ in range(len(self.team))]  # np.zeros([len(self.predators), 2],dtype=int)

        self.dead = [False] * len(self.team)

        for i in range(len(self.team)):
            x, y = self.get_empty_location()
            self.map[x, y] = AGENT
            self.team[i][0] = x
            self.team[i][1] = y

        self.turn = 0

        self.terminal = False
        self.playingRPS = self.checkIfPlayingRPS()

        self.paint_map()

    # moves agents. if they are adjacent, then they play RPS
    def move(self, id_, action):
        self.turn += 1
        self.rewards = [0] * len(self.rewards)
        #print("moving",id_,self.terminal, self.playingRPS)
        if self.playingRPS:
            #print("playing rps")
            self.terminal=True
            self.rps_actions[id_] = action

            self.rewards[0] = self.R1[self.rps_actions[0]][self.rps_actions[1]]
            self.rewards[1] = self.R2[self.rps_actions[0]][self.rps_actions[1]]
            """if self.rps_actions[0] == 0 or self.rps_actions[1] == 0:
                self.rewards[0] = -1
                self.rewards[1] = -1
            elif (self.rps_actions[0] == 1 and self.rps_actions[1] == 2) or \
               (self.rps_actions[0] == 2 and self.rps_actions[1] == 3) or \
               (self.rps_actions[0] == 3 and self.rps_actions[1] == 1):
                self.rewards[0] = 1
                self.rewards[1] = 3
            elif (self.rps_actions[0] == 1 and self.rps_actions[1] == 3) or \
                 (self.rps_actions[0] == 2 and self.rps_actions[1] == 1) or \
                 (self.rps_actions[0] == 3 and self.rps_actions[1] == 2):
                self.rewards[0] = 3
                self.rewards[1] = 1
            else:
                self.rewards[0] = 2
                self.rewards[1] = 2"""

            # we reward 1,2,3 in RPS instead of -1,0,1 so that expected reward is positive and they agents actually want to play the game

            return

        mod_x, mod_y = self.get_mods(action)

        new_x = int((self.team[id_][0] + mod_x) % self.width)
        new_y = int((self.team[id_][1] + mod_y) % self.height)

        if self.map[new_x, new_y]!=AGENT:
            self.map[self.team[id_][0], self.team[id_][1]] = np.random.random() / 2
            self.team[id_][0] = new_x
            self.team[id_][1] = new_y
            #print("move", id_, new_x, new_y)
            self.map[self.team[id_][0], self.team[id_][1]] = AGENT
        #else:
            #print("no move", id_, new_x, new_y)

    def get_empty_location(self):
        x, y = np.random.randint(0, self.width - 1), np.random.randint(0, self.height)
        while self.map[x, y] == 1:
            x, y = np.random.randint(0, self.width - 1), np.random.randint(0, self.height)
        return x, y

    def get_mods(self, action):
        # moves = ["(move none)", "(move south)", "(move north)", "(move west)", "(move east)"]

        mod_x = 0
        mod_y = 0
        if action == 0:
            mod_x = 0
            mod_y = 1
        elif action == 1:
            mod_x = 0
            mod_y = -1
        elif action == 2:
            mod_x = -1
            mod_y = 0
        elif action == 3:
            mod_x = 1
            mod_y = 0
        return mod_x, mod_y

    def print_map(self):
        self.map = np.zeros([self.width, self.height])

        for predator in self.team:
            self.map[predator[0], predator[1]] = 1

        for y in range(self.height):
            for x in range(self.width):
                print(int(self.map[x][y]), end='')
            print()
        print()

    def get_state_global(self):
        self.paint_map()
        return np.transpose(self.map)

    def checkIfPlayingRPS(self):
        if (abs(self.team[0][0] - self.team[1][0]) + abs(self.team[0][1] - self.team[1][1]) == 1):
            #print("next to each other")
            return True
        if abs(self.team[0][0] - self.team[1][0]) == self.width-1 and abs(self.team[0][1] - self.team[1][1]) == 0:
            #print("side walls", self.team[0][0], self.team[1][0], self.team[0][1], self.team[1][1])
            return True
        if abs(self.team[0][0] - self.team[1][0]) == 0 and abs(self.team[0][1] - self.team[1][1]) == self.height-1:
            #print("top walls")
            return True
        return False

    def get_state(self, id_):
        if self.checkIfPlayingRPS():
            self.map = np.full((self.width, self.height), -1)
            self.playingRPS = True
        #else:
        #    print(self.team[0][0],self.team[1][0],self.team[0][1],self.team[1][1])

        base_x, base_y = self.team[id_]

        state = np.roll(np.roll(np.transpose(self.map), int(self.height / 2) - base_y, axis=0),
                        int(self.width / 2) - base_x, axis=1)

        return state, self.terminal, self.rewards[id_]
