from gym.envs.gym_maze.envs import AbstractMaze

import numpy as np
#Mapping (Indicator - Reward)
#5 - 3
#6 - 4

class IMaze3(AbstractMaze):
    def __init__(self):
        #Position of indicator. (x,y)
        self.ipos = [1, 1]
        #Start position of agent.
        self.apos = [1,3]
        super(IMaze3, self).__init__(np.matrix([
            [2, 2, 2, 2, 2, 2, 2],
            [2, 5, 2, 2, 2, 3, 2],
            [2, 1, 1, 1, 1, 1, 2],
            [2, 1, 2, 2, 2, -3, 2],
            [2, 2, 2, 2, 2, 2, 2]
        ]))