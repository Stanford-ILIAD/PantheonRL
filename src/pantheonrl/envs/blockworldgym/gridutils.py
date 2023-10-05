""" Functions to deal with generating a grid world and gravity. """
import numpy as np

HORIZONTAL = 0
VERTICAL = 1


def generate_random_world(width, num_blocks, num_colors, np_random):
    """Generates a random world"""
    gridworld = np.zeros((width, width))
    blocks_placed = 0
    while blocks_placed < num_blocks:
        if drop_random(width, gridworld, num_colors, np_random) != -1:
            blocks_placed += 1
    return gridworld


def drop_random(width, gridworld, num_colors, np_random):
    """Drops a block randomly"""
    orientation = np_random.integers(0, 2)
    if orientation == HORIZONTAL:
        x = np_random.integers(
            0, width - 1
        )  # can't drop at the last coordinate if horizontal
    else:
        x = np_random.integers(0, width)
    y = gravity(gridworld, orientation, x)
    if y == -1:
        return -1  # error
    color = np_random.integers(0, num_colors) + 1
    place(gridworld, x, y, color, orientation)
    return 0


def place(gridworld, x, y, color, orientation):
    """Places block into the grid"""
    gridworld[y][x] = color
    if orientation == HORIZONTAL:
        gridworld[y][x + 1] = color
    if orientation == VERTICAL:
        gridworld[y + 1][x] = color


def gravity(gridworld, orientation, x):
    """Runs a timestep of gravity simulation"""
    # check if placeable
    if gridworld[0][x] != 0:
        return -1
    if (orientation == HORIZONTAL and gridworld[0][x + 1] != 0) or (
        orientation == VERTICAL and gridworld[1][x] != 0
    ):
        return -1
    for y in range(len(gridworld)):
        # this is the final position if it hits something
        # (there's something or a floor right below it)
        if orientation == HORIZONTAL:
            if (
                y == len(gridworld) - 1
                or gridworld[y + 1][x] != 0
                or gridworld[y + 1][x + 1] != 0
            ):
                return y
        if orientation == VERTICAL:
            if y == len(gridworld) - 2 or gridworld[y + 2][x] != 0:
                return y
    return -1  # shouldn't be able to reach here


def matches(grid1, grid2):
    """Check if two grids match"""
    # number of nonzero elements in the same place
    return np.count_nonzero(grid1 == grid2)
