#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym.envs.gym_maze import Maze, DARK_MAPPING, PATH_MAPPING, WALL_MAPPING, \
                                        REWARD_MAPPING, ACTION_LOOKUP, INDICATOR_MAPPING, \
                                        ANIMAT_MARKER

from gym.envs.gym_maze.utils import get_all_possible_transitions

import numpy as np
import logging
import random
import sys


logger = logging.getLogger(__name__)

class AbstractMaze(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, matrix):
        self.maze = Maze(matrix)
        self.maze_dim = matrix.shape

        self.give_position = False

        self.pos_x = None
        self.pos_y = None
        self.previous_pos_x = None
        self.previous_pos_y = None

        self.ival = -1
        if hasattr(self, 'ipos'):
            print self.ipos
            self.ival = self.maze.matrix[self.ipos[1],self.ipos[0]]

        #Minecraft paper stops an episode after 50 steps
        self.steps_taken = 0 

        nb_actions = 4
        self.action_space = spaces.Discrete(nb_actions)
        #self.observation_space = spaces.Discrete(8)
        observation_size = self.maze.max_x*self.maze.max_y
        self.observation_space = spaces.Discrete(observation_size)

    def _step(self, action):
        previous_observation = self._observe()
        self._take_action(action, previous_observation)

        observation = self._observe()

        if self.give_position:
            observation[0] = np.float32(observation[0])
        else:
            observation = np.float32(observation)

        #observation.reshape((-1,)) #Flatten observation?
        #print observation
        reward = self._get_reward()
        episode_over = self._is_over()

        self.steps_taken += 1

        return observation, reward, episode_over, {}

    def _reset(self):
        logger.debug("Resetting the environment")
        self._insert_animat()

        if hasattr(self, 'ipos'):
            choice = random.choice(INDICATOR_MAPPING)
            self.maze.matrix[self.ipos[1],self.ipos[0]] = choice
            self.ival = choice

        return self._observe()

    def _render(self, mode='human', close=False):
        if close:
            return

        logging.debug("Rendering the environment")
        if mode == 'human':
            outfile = sys.stdout
            outfile.write("\n")

            situation = np.copy(self.maze.perception(self.pos_x, self.pos_y))
            '''
            situation = np.array([
                [situation[7], situation[0], situation[1]],
                [situation[6], ANIMAT_MARKER, situation[2]],
                [situation[5], situation[4], situation[3]]], dtype=np.int8)
            '''
            situation[self.pos_y, self.pos_x] = ANIMAT_MARKER

            for row in situation:
                outfile.write("".join(self._render_element(el) for el in row))
                outfile.write("\n")
        else:
            super(AbstractMaze, self).render(mode=mode)

    def _observe(self):
        if self.give_position:
            return [self.maze.perception(self.pos_x, self.pos_y), np.array([self.pos_x, self.pos_y])]
        else:
            return self.maze.perception(self.pos_x, self.pos_y)

    def rewardCorrect(self, ival, rval):
        '''Returns if the reward chosen is correct based on the
        value of the indicator.
        Returns true if the index of the indicator value is the 
        same as the index of the reward value, false otherwise.
        '''

        i_idx = INDICATOR_MAPPING.index(ival)
        r_idx = REWARD_MAPPING.index(rval)

        return i_idx == r_idx

    def _get_reward(self):
        #Agent is on a goal point
        if self.maze.is_reward(self.pos_x, self.pos_y):
            rval = self.maze.matrix[self.pos_y,self.pos_x]
            if(hasattr(self, 'ipos') and not self.rewardCorrect(self.ival,rval)):
                return -0.4
            else:
                return 1.   

        #Agent is not at a goal point.
        step_reward = -0.02
        collision_multiplier = 1

        #More negative reinforcement for crashing into walls
        if self.previous_pos_x == self.pos_x and self.previous_pos_y == self.pos_y:
            step_reward *= collision_multiplier

        return step_reward
        #return 0

    def _is_over(self):
        #return self.maze.is_reward(self.pos_x, self.pos_y)
        if(self.maze.is_reward(self.pos_x, self.pos_y) or self.steps_taken >= 50):
            self.steps_taken = 0
            ''' TODO: Verify this
            if hasattr(self, 'ipos'):
                choice = random.choice(INDICATOR_MAPPING)
                self.maze.matrix[self.ipos[1],self.ipos[0]] = choice
                self.ival = choice
            '''
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
        #ne = (self.pos_x+1, self.pos_y-1)
        e = (self.pos_x+1, self.pos_y)
        #se = (self.pos_x+1, self.pos_y+1)
        s = (self.pos_x, self.pos_y+1)
        #sw = (self.pos_x-1, self.pos_y+1)
        w = (self.pos_x-1, self.pos_y)
        #nw = (self.pos_x-1, self.pos_y-1)

        #Save current position before moving
        self.previous_pos_x, self.previous_pos_y = (self.pos_x, self.pos_y)

        if action_type == "N" and not self.maze.is_wall(n[0],n[1]):
            self.pos_y -= 1
            animat_moved = True
        '''
        if action_type == 'NE' and not self.maze.is_wall(ne[0],ne[1]):
            self.pos_x += 1
            self.pos_y -= 1
            animat_moved = True
        '''
        if action_type == "E" and not self.maze.is_wall(e[0],e[1]):
            self.pos_x += 1
            animat_moved = True
        '''
        if action_type == 'SE' and not self.maze.is_wall(se[0],se[1]):
            self.pos_x += 1
            self.pos_y += 1
            animat_moved = True
        '''
        if action_type == "S" and not self.maze.is_wall(s[0],s[1]):
            self.pos_y += 1
            animat_moved = True
        '''
        if action_type == 'SW' and not self.maze.is_wall(sw[0],sw[1]):
            self.pos_x -= 1
            self.pos_y += 1
            animat_moved = True
        '''
        if action_type == "W" and not self.maze.is_wall(w[0],w[1]):
            self.pos_x -= 1
            animat_moved = True
        '''
        if action_type == 'NW' and not self.maze.is_wall(nw[0],nw[1]):
            self.pos_x -= 1
            self.pos_y -= 1
            animat_moved = True
        '''
        return animat_moved

    def _insert_animat(self):
        starting_position = None

        if(hasattr(self, 'apos')):
            starting_position = self.apos
        else:
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
        elif el == REWARD_MAPPING[0]:
            return utils.colorize('$', 'red')
        elif el == REWARD_MAPPING[1]:
            return utils.colorize('$', 'blue')
        elif el == INDICATOR_MAPPING[0]:
            return utils.colorize('#', 'red')
        elif el == INDICATOR_MAPPING[1]:
            return utils.colorize('#', 'blue')
        elif el == ANIMAT_MARKER:
            return utils.colorize('A', 'red')
        elif el == DARK_MAPPING:
            return utils.colorize(u'▒', 'gray')
        else:
            return utils.colorize(el, 'cyan')
