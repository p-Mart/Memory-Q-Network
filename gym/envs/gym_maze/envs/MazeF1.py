from gym.envs.gym_maze.envs import AbstractMaze

import numpy as np


class MazeF1(AbstractMaze):
    def __init__(self):
        super(MazeF1, self).__init__(np.matrix([
            [1, 1, 1, 1],
            [1, 0, 3, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ]))
