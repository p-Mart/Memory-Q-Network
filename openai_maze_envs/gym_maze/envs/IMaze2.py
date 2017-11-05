from gym_maze.envs import AbstractMaze

import numpy as np
#Mapping (Indicator - Reward)
#5 - 3
#6 - 4

class IMaze2(AbstractMaze):
    def __init__(self):
        #Position of indicator. (x,y)
        self.ipos = [1, 1]
        #Start position of agent.
        self.apos = [1,3]
        super(IMaze2, self).__init__(np.matrix([
            [1, 1, 1, 1, 1, 1],
            [1, 5, 1, 1, 3, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 4, 1],
            [1, 1, 1, 1, 1, 1]
        ]))
