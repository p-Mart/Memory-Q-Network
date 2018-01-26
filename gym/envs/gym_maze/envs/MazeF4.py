from gym.envs.gym_maze.envs import AbstractMaze

import numpy as np


class MazeF4(AbstractMaze):
    def __init__(self):
        super(MazeF4, self).__init__(np.matrix([
            [2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 3, 2],
            [2, 1, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 2, 2],
            [2, 1, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ]))
