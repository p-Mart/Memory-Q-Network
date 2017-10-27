#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_maze import Maze, DARK_MAPPING, PATH_MAPPING, WALL_MAPPING, \
                                        REWARD_MAPPING, ACTION_LOOKUP

from gym_maze.utils import get_all_possible_transitions

import numpy as np
import logging
import random
import sys


logger = logging.getLogger(__name__)

ANIMAT_MARKER = 5


class AbstractMaze(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, matrix):
        self.maze = Maze(matrix)
        self.pos_x = None
        self.pos_y = None
        #Minecraft paper stops an episode after 50 steps
        self.steps_taken = 0 
        self.action_space = spaces.Discrete(8)
        #self.observation_space = spaces.Discrete(8)
        observation_size = self.maze.max_x*self.maze.max_y
        self.observation_space = spaces.Discrete(observation_size)

    def _step(self, action):
        previous_observation = self._observe()
        self._take_action(action, previous_observation)

        observation = self._observe()
        #observation.reshape((-1,)) #Flatten observation?
        #print observation
        reward = self._get_reward()
        episode_over = self._is_over()

        self.steps_taken += 1

        return observation, reward, episode_over, {}

    def _reset(self):
        logger.debug("Resetting the environment")
        self._insert_animat()
        return self._observe()

    def _render(self, mode='human', close=False):
        if close:
            return

        logging.debug("Rendering the environment")
        if mode == 'human':
            outfile = sys.stdout
            outfile.write("\n")

            situation = np.copy(self.maze.perception(self.pos_x, self.pos_y))
            situation[self.pos_y, self.pos_x] = ANIMAT_MARKER

            for row in situation:
                outfile.write("".join(self._render_element(el) for el in row))
                outfile.write("\n")
        else:
            super(AbstractMaze, self).render(mode=mode)

    def _observe(self):
        return self.maze.perception(self.pos_x, self.pos_y)

    def _get_reward(self):
        if self.maze.is_reward(self.pos_x, self.pos_y):
            return 1.

        #Modification as per Minecraft paper:
        return -0.04
        #return 0

    def _is_over(self):
        #return self.maze.is_reward(self.pos_x, self.pos_y)
        if(self.maze.is_reward(self.pos_x, self.pos_y) or self.steps_taken >= 50):
            self.steps_taken = 0
            return True
        else:
            return False

    def get_all_possible_transitions(self):
        """
        Debugging only
        
        :return: 
        """
        return get_all_possible_transitions(self)

    def _take_action(self, action, observation):
        """Executes the action inside the maze"""
        animat_moved = False
        action_type = ACTION_LOOKUP[action]

        n = (self.pos_x, self.pos_y - 1)
        ne = (self.pos_x+1, self.pos_y-1)
        e = (self.pos_x+1, self.pos_y)
        se = (self.pos_x+1, self.pos_y+1)
        s = (self.pos_x, self.pos_y+1)
        sw = (self.pos_x-1, self.pos_y+1)
        w = (self.pos_x-1, self.pos_y)
        nw = (self.pos_x-1, self.pos_y-1)


        if action_type == "N" and not self.maze.is_wall(n[0],n[1]):
            self.pos_y -= 1
            animat_moved = True

        if action_type == 'NE' and not self.maze.is_wall(ne[0],ne[1]):
            self.pos_x += 1
            self.pos_y -= 1
            animat_moved = True

        if action_type == "E" and not self.maze.is_wall(e[0],e[1]):
            self.pos_x += 1
            animat_moved = True

        if action_type == 'SE' and not self.maze.is_wall(se[0],se[1]):
            self.pos_x += 1
            self.pos_y += 1
            animat_moved = True

        if action_type == "S" and not self.maze.is_wall(s[0],s[1]):
            self.pos_y += 1
            animat_moved = True

        if action_type == 'SW' and not self.maze.is_wall(sw[0],sw[1]):
            self.pos_x -= 1
            self.pos_y += 1
            animat_moved = True

        if action_type == "W" and not self.maze.is_wall(w[0],w[1]):
            self.pos_x -= 1
            animat_moved = True

        if action_type == 'NW' and not self.maze.is_wall(nw[0],nw[1]):
            self.pos_x -= 1
            self.pos_y -= 1
            animat_moved = True

        return animat_moved

    def _insert_animat(self):
        possible_coords = self.maze.get_possible_insertion_coordinates()

        starting_position = random.choice(possible_coords)
        self.pos_x = starting_position[0]
        self.pos_y = starting_position[1]

    @staticmethod
    def is_wall(perception):
        return perception == WALL_MAPPING

    @staticmethod
    def _render_element(el):
        if el == WALL_MAPPING:
            return utils.colorize(u'█', 'white')
        elif el == PATH_MAPPING:
            return utils.colorize('.', 'white')
        elif el == REWARD_MAPPING:
            return utils.colorize('$', 'green')
        elif el == ANIMAT_MARKER:
            return utils.colorize('A', 'red')
        elif el == DARK_MAPPING:
            return utils.colorize(u'▒', 'gray')
        else:
            return utils.colorize(el, 'cyan')
