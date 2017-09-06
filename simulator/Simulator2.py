import numpy as np
import pygame
from time import sleep

BLACK = np.array((0, 0, 0),dtype=np.uint8)           # obstacle
WHITE = np.array((255, 255, 255),dtype=np.uint8)     # clear
GREEN = np.array((0, 255, 0),dtype=np.uint8)         # prey
BLUE = np.array((0, 0, 255),dtype=np.uint8)          # predator
colors = [WHITE, BLUE, GREEN, BLACK]
WIDTH = 10
HEIGHT = 10
MARGIN = 1

WALL = 1
PREY = 0.666
PREDATOR = 0.333


class PursuitSim(object):
    def __init__(self, width, height, predators, prey, walls=3, wall_radius=0, gui=False): # 10 1
        self.turn = 0

        self.imageCounter = 0
        self.saveImage = False

        self.width = width
        self.height = height

        self.map = np.zeros([width, height])
        self.predators = [[0, 0] for _ in range(predators)]  # np.zeros([predators, 2], dtype=int)
        self.prey = [[0, 0] for _ in range(prey)]  # np.zeros([prey, 2], dtype=int)
        self.walls = [[0, 0] for _ in range(walls)]  # np.zeros([walls, 2], dtype=int)
        self.wall_radius = wall_radius

        # self.preyCollisions = [False]*prey
        self.preyCaught = [False] * prey

        self.predatorReward = 0
        self.preyReward = 0

        self.terminal_predator = False
        self.terminal_prey = False

        self.reset()

        self.gui = gui
        if gui:
            pygame.init()
            self.screen = pygame.display.set_mode([(WIDTH + MARGIN) * width, (HEIGHT + MARGIN) * height])
            pygame.display.set_caption("Pursuit")
            self.paint_map()
            # self.prev_debug_flag=0

    def paint_map(self):
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
        # print("reset_map")
        self.map = np.zeros([self.width, self.height])
        self.predators = [[0, 0] for _ in range(len(self.predators))]  # np.zeros([len(self.predators), 2],dtype=int)
        self.prey = [[0, 0] for _ in range(len(self.prey))]  # np.zeros([len(self.prey), 2],dtype=int)
        self.walls = [[0, 0] for _ in range(len(self.walls))]  # np.zeros([len(self.walls), 2],dtype=int)

        self.preyCaught = [False] * len(self.prey)

        for i in range(len(self.walls)):
            x, y = np.random.randint(0, self.width - 1), np.random.randint(0, self.height - 1)
            while self.map[x - self.wall_radius - 1, y - self.wall_radius - 1] != 0 or \
                            self.map[x - self.wall_radius - 1, (y + self.wall_radius + 1) % self.height] != 0 or \
                            self.map[(x + self.wall_radius + 1) % self.width, y - self.wall_radius - 1] != 0 or \
                            self.map[(x + self.wall_radius + 1) % self.width, (
                                    y + self.wall_radius + 1) % self.height] != 0:
                x, y = np.random.randint(0, self.width - 1), np.random.randint(0, self.height - 1)
            for j in range(-self.wall_radius, self.wall_radius + 1):
                for k in range(-self.wall_radius, self.wall_radius + 1):
                    self.map[x + j, y + k] = WALL
            self.walls[i][0] = x
            self.walls[i][1] = y

        for i in range(len(self.predators)):
            x, y = self.get_empty_location()
            self.map[x, y] = PREDATOR
            self.predators[i][0] = x
            self.predators[i][1] = y

        for i in range(len(self.prey)):
            x, y = self.get_empty_location()
            self.map[x, y] = PREY
            self.prey[i][0] = x
            self.prey[i][1] = y

        self.turn = 0

        self.terminal_predator = False
        self.terminal_prey = False

    # removes predators from map and assigns new (possibly not-valid) positions
    def movePredator(self, id_, action):
        self.turn += 1
        self.predatorReward = 0

        self.map[self.predators[id_][0], self.predators[id_][1]] = 0

        mod_x, mod_y = self.get_mods(action)

        new_x = int((self.predators[id_][0] + mod_x) % self.width)
        new_y = int((self.predators[id_][1] + mod_y) % self.height)
        if self.in_wall(new_x, new_y):
            return

        self.predators[id_][0] = new_x
        self.predators[id_][1] = new_y

    # removes prey from map and assigns new (possibly not-valid) positions
    def movePrey(self, id_, action):
        self.preyReward = 0
        self.turn += 1
        if self.preyCaught[id_]:
            return

        self.map[self.prey[id_][0], self.prey[id_][1]] = 0

        mod_x, mod_y = self.get_mods(action)

        new_x = int((self.prey[id_][0] + mod_x) % self.width)
        new_y = int((self.prey[id_][1] + mod_y) % self.height)
        if self.in_wall(new_x, new_y):
            return

        self.prey[id_][0] = new_x
        self.prey[id_][1] = new_y

    def get_empty_location(self):
        x, y = np.random.randint(0, self.width - 1), np.random.randint(0, self.height - 1)
        while self.map[x, y] != 0:
            x, y = np.random.randint(0, self.width - 1), np.random.randint(0, self.height - 1)
        return x, y

    def in_wall(self, x, y):
        return self.map[x][y] == WALL

    def get_mods(self, action):
        # moves = ["(move none)", "(move south)", "(move north)", "(move west)", "(move east)"]

        mod_x = 0
        mod_y = 0
        if action == 0:
            mod_x = 0
            mod_y = 0
        elif action == 1:
            mod_x = 0
            mod_y = 1
        elif action == 2:
            mod_x = 0
            mod_y = -1
        elif action == 3:
            mod_x = -1
            mod_y = 0
        elif action == 4:
            mod_x = 1
            mod_y = 0
        return mod_x, mod_y

    # assigns predators to valid positions and penalizes for collisions
    def checkCollisionsPredator(self):
        colliding = [i for i, x in enumerate(self.predators) if self.predators.count(x) > 1]  # O(n^2)

        for col_index in colliding:
            self.predatorReward -= 1
            self.predators[col_index] = list(self.get_empty_location())

    # assigns prey to valid positions and penalizes for collisions
    def checkCollisionsPrey(self):
        colliding = [i for i, x in enumerate(self.prey) if not self.preyCaught[i] and self.prey.count(x) > 1]  # O(n^2)

        for col_index in colliding:
            self.preyReward -= 1
            self.prey[col_index] = list(self.get_empty_location())

    # places prey on the map if they were not eaten
    def checkPredatorCaughtPrey_prey(self):
        eaten = [i for i, x in enumerate(self.prey) if not self.preyCaught[i] and self.predators.count(x) > 0]  # O(n^2)

        for col_index in eaten:
            self.preyReward -= 1
            self.predatorReward += 1
            self.preyCaught[col_index] = True

        for x, eaten in zip(self.prey, self.preyCaught):
            if not eaten:
                self.map[int(x[0]), int(x[1])] = PREY

        now_terminal = np.sum(self.preyCaught) == len(self.prey)
        self.terminal_predator = self.terminal_predator or now_terminal
        self.terminal_prey = self.terminal_prey or now_terminal
        self.paint_map()

    # places predators on the map
    def checkPredatorCaughtPrey_predator(self):
        eaten = [i for i, x in enumerate(self.prey) if not self.preyCaught[i] and self.predators.count(x) > 0]  # O(n^2)

        for col_index in eaten:
            self.preyReward -= 1
            self.predatorReward += 1
            self.preyCaught[col_index] = True

        for x in self.predators:
            # print(self.predators)
            self.map[int(x[0]), int(x[1])] = PREDATOR

        now_terminal = np.sum(self.preyCaught) == len(self.prey)
        self.terminal_predator = self.terminal_predator or now_terminal
        self.terminal_prey = self.terminal_prey or now_terminal
        self.paint_map()

    def print_map(self):
        self.map = np.zeros([self.width, self.height])

        for predator in self.predators:
            self.map[predator[0], predator[1]] = 1
        for i in range(len(self.prey)):
            if not self.preyCaught[i]:
                self.map[self.prey[i][0], self.prey[i][1]] = 2

        for y in range(self.height):
            for x in range(self.width):
                print(int(self.map[x][y]), end='')
            print()
        print()

    def get_state_global(self):
        return np.transpose(self.map)

    def get_state_predator(self, id_):
        base_x, base_y = self.predators[id_]

        state = np.roll(np.roll(np.transpose(self.map), int(self.height / 2) - base_y, axis=0),
                        int(self.width / 2) - base_x, axis=1)

        # self.paint_map()

        return state, self.terminal_predator, self.predatorReward

    def get_state_prey(self, id_):
        base_x, base_y = self.prey[id_]

        state = np.roll(np.roll(np.transpose(self.map), int(self.height / 2) - base_y, axis=0),
                        int(self.width / 2) - base_x, axis=1)

        return state, self.preyCaught[id_], self.terminal_prey, self.preyReward
