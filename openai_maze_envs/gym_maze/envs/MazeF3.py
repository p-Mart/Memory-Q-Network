from gym_maze.envs import AbstractMaze

import numpy as np


class MazeF3(AbstractMaze):
    def __init__(self):
        super(MazeF3, self).__init__(np.matrix([
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 3, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]))