from random import choice

import logging

import gym

import pygame
from pygame.locals import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

maze = gym.make('IMaze6-v0')

possible_actions = list(range(8))
#transitions = maze.env.get_all_possible_transitions()

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Pygame Keyboard Test')
pygame.mouse.set_visible(0)

action = -1
t = 0
for i_episode in range(20):
    observation = maze.reset()

    while True:
        maze.render()
        #Block until key input
        while(action == -1):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if (event.key == K_UP):
                        action = 0
                    elif(event.key == K_RIGHT):
                        action = 1
                    elif(event.key == K_DOWN):
                        action = 2
                    elif(event.key == K_LEFT):
                        action = 3

        #action = choice(possible_actions)
        logger.info("\t\tExecuted action: [{}]".format(action))
        observation, reward, done, info = maze.step(action)
        t += 1
        action = -1
        if done:
            logger.info("Episode finished after {} timesteps.".format(t + 1))
            logger.info("Last reward: {}".format(reward))
            break

logger.info("Finished")
