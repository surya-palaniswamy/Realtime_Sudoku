#!/usr/bin/env python3

class Solver:
    def __init__(self, grid):
        """
            >The class is instantiated with the sudoku grid that was presented to the solver.
        """
        self.grid = grid
        self.truth = list(map(bool,val) for val in self.grid)
        self.sum = sum(list(sum(self.grid, [])))

    
    def display(self):
        for val in self.grid:
            print(val)

    def get_row(self,row):
        return self.grid[row]

    def get_column(self,col):
        return [row[col] for row in self.grid]

    def get_neighbours(self,row,col):
        box_row = (row//3)*3
        box_col = (col//3)*3
        box = [val[box_col:box_col +3] for val in self.grid[box_row:box_row+3]]
        box = ([box[i][j] for i in range(3) for j in range(3)])
        return box

    def find_possibilites(self,row,col):
        options = list(range(1,10))
        row_vals = self.get_row(row)
        col_vals = self.get_column(col)
        neighbour_vals = self.get_neighbours(row,col)
        for val in row_vals + col_vals + neighbour_vals:
            if val in options:
                options.remove(val)
        return options

    def fix_number(self,row,col,val):
        self.grid[row][col] = val
        self.truth[row][col] = True
        self.sum+=val
    
    def undo_number(self,row,col,val):
        self.grid[row][col] = 0
        self.truth[row][col] = False
        self.sum-=val
    
    def grid_solve(self):
        if self.sum == 405:
            self.display()
            return True
        for i in range(9):
            for j in range(9):
                if not self.truth[i][j]:
                    options = self.find_possibilites(i,j)
                    for val in options:
                        self.fix_number(i,j,val)
                        if self.grid_solve():
                            return True
                        else:
                            self.undo_number(i,j,val)
                    return False