from gym.envs.gym_maze.envs import AbstractMaze

import numpy as np


class MazeF4(AbstractMaze):
    def __init__(self):
        super(MazeF4, self).__init__(np.matrix([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 3, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]))
