import numpy as np
from game import Game

def op_player(s):
    if list(s) == [0,0,0,0,0,0,0,0,0]:
        return 0
    if list(s) == [1,0,0,0,0,0,0,0,0] or list(s) == [0,0,1,0,0,0,0,0,0] or list(s) == [0,0,0,0,0,0,1,0,0] or list(s) == [0,0,0,0,0,0,0,0,1]:
        return 4
    elif list(s) == [0,1,0,0,0,0,0,0,0] or list(s) == [0,0,0,1,0,0,0,0,0] or list(s) == [0,0,0,0,0,1,0,0,0] or list(s) == [0,0,0,0,0,0,0,1,0]:
        return 4
    elif list(s) == [0,0,0,0,1,0,0,0,0]:
        return 0

    board = [list(a) for a in s.reshape([3,3])]
    boardT = [list(a) for a in s.reshape([3,3]).T]

    if [1,1,0] in board:
        return 3*board.index([1,1,0]) + 2
    elif [1,1,0] in boardT:
        return 6 + boardT.index([1,1,0]) 
    if [1,0,1] in board:
        return 3*board.index([1,0,1]) + 1
    elif [1,0,1] in boardT:
        return 3 + boardT.index([1,0,1])
    if [0,1,1] in board:
        return 3*board.index([0,1,1])
    elif [0,1,1] in boardT:
        return boardT.index([0,1,1])

    if abs(s[0]) == 1 and abs(s[4]) == 1 and abs(s[8]) == 0:
        return 8
    elif abs(s[0]) == 1 and abs(s[4]) == 0 and abs(s[8]) == 1:
        return 4
    elif abs(s[0]) == 0 and abs(s[4]) == 1 and abs(s[8]) == 1:
        return 0

    if abs(s[2]) == 1 and abs(s[4]) == 1 and abs(s[6]) == 0:
        return 6
    elif abs(s[2]) == 1 and abs(s[4]) == 0 and abs(s[6]) == 1:
        return 4
    elif abs(s[2]) == 0 and abs(s[4]) == 1 and abs(s[6]) == 1:
        return 2

    if [-1,-1,0] in board:
        return 3*board.index([-1,-1,0]) + 2
    elif [-1,-1,0] in boardT:
        return 6 + boardT.index([-1,-1,0])
    if [-1,0,-1] in board:
        return 3*board.index([-1,0,-1]) + 1
    elif [-1,0,-1] in boardT:
        return 3 + boardT.index([-1,0,-1])
    if [0,-1,-1] in board:
        return 3*board.index([0,-1,-1])
    elif [0,-1,-1] in boardT:
        return boardT.index([0,-1,-1])

    else:
        return random_player(s)

def random_player(s):
    moves = Game.get_valid_actions(s)
    move = np.random.choice(moves)
    return move

