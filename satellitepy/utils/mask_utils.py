import random

def find_random_connected_block(grid):
    height = len(grid)
    width = len(grid[0])

    while True:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        if grid[y][x]:
            return x, y

def calculate_adjacent_coordinates(x, y, direction):
    if direction == 'up':
        return x, y - 1
    elif direction == 'down':
        return x, y + 1
    elif direction == 'left':
        return x - 1, y
    elif direction == 'right':
        return x + 1, y

def is_valid_coordinates(x, y, width, height):
    return 0 <= x < width and 0 <= y < height 