import time
import numpy as np
import random

def move_piece(game, pos, lower_pos):
    for i in pos:
        game[i[0], i[1]] = 0
        i[0] += 1
    for i in pos:
        game[i[0], i[1]] = 1
    for i in lower_pos:
        i[0] += 1


def create_piece(game):
    num = random.randrange(1)
    pos = []
    lower_pos = []
    if num == 0:
        pos = [[0, 0], [0, 1], [0, 2], [0, 3]]
        lower_pos = [[0, 0], [0, 1], [0, 2], [0, 3]]
        game[0, 0] = 1
        game[0, 1] = 1
        game[0, 2] = 1
        game[0, 3] = 1
    elif num == 1:
        pos = [[0, 0], [0, 1], [1, 0], [1, 1]]
        lower_pos = [[1, 0], [1, 1]]
        game[0, 0] = 1
        game[0, 1] = 1
        game[1, 0] = 1
        game[1, 1] = 1
    return pos, lower_pos


def piece_landed(game, lower_pos):
    for i in lower_pos:
        if i[0]+1 == game.shape[0] or game[i[0]+1, i[1]] == 1:
            return True


def game_over(game):
    for col in range(game.shape[1]):
        if(game[1, col] == 1):
            return True


def score(game):
    reward = 0
    for row in range(game.shape[0]):
        scored_row = True
        for col in range(game.shape[1]):
            if(game[row, col] == 0):
                scored_row = False
                break
        if(scored_row):
            reward += 1
            game[row, :] = 0
            for i in reversed(range(0, row)):
                for j in range(game.shape[1]):
                    if(game[i, j] == 1):
                        game[i, j] = 0
                        game[i+1, j] = 1
    return reward


def move_left(game, pos, lower_pos):
    possible = True
    for i in pos:
        if i[1] == 0:
            possible = False
    if possible:
        for i in pos:
            game[i[0], i[1]] = 0
        for i in pos:
            i[1] -= 1
            game[i[0], i[1]] = 1
        for i in lower_pos:
            i[1] -= 1


def move_right(game, pos, lower_pos):
    possible = True
    for i in pos:
        if i[1] == game.shape[1]-1:
            possible = False
    if possible:
        for i in pos:
            game[i[0], i[1]] = 0
        for i in pos:
            i[1] += 1
            game[i[0], i[1]] = 1
        for i in lower_pos:
            i[1] += 1


game = np.zeros((24, 12))
game[game.shape[0]-1] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
game[game.shape[0]-2] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
pos, lower_pos = create_piece(game)

while(True):
    print(f"----------\n{game}\n----------\n")
    if(piece_landed(game, lower_pos)):
        if(game_over(game)):
            game = np.zeros((24, 10))
        score(game)
        pos, lower_pos = create_piece(game)
    move_right(game, pos, lower_pos)
    move_piece(game, pos, lower_pos)
    time.sleep(0.5)
