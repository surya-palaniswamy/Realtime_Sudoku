#!/usr/bin/env python3

import numpy as np
from sudoku_solver_functions import Solver
from timeit import default_timer as timer


puzzle = [[4, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 9, 8],
        [3, 0, 0, 0, 8, 2, 4, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 8, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 6, 7, 0],
        [0, 5, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 9, 0, 7],
        [6, 4, 0, 3, 0, 0, 0, 0, 0]]

puzzle2 = [[0, 6, 0, 4, 0 ,0, 0, 7, 0],
                 [0, 8, 0, 0, 0, 0, 0, 2, 9],
                 [0, 7, 0, 0, 2, 0, 5, 0, 0],
                 [0, 0, 5, 6, 0, 0, 0, 0, 4],
                 [9, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 3],
                 [0, 0, 4, 1, 0, 0, 0, 0, 0],
                 [8, 0, 0, 0, 9, 0, 0, 0, 0],
                 [0, 0, 0, 0, 8, 0, 1, 0, 6]]
call = Solver(puzzle)
start = timer()
call.grid_solve()
end = timer()
print('Normal Backtracking: ',end - start,'\n')