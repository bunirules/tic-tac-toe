from nnet import Network
from game import Game
from mcts import MCTS
import players
import numpy as np

def play_OP(net, search_sims, c_puct):
    # i = np.random.choice([0,1])
    p_ai = "X" # ["X", "O"][i]
    player = "O" # ["X", "O"][1-i]

    game = Game()
    s = game.get_initial_state(player=p_ai)

    current_player = "X"

    mcts = MCTS()

    while True:
        if player == current_player:
            move = players.op_player(s)
            s = game.next_state(s, move)
        elif p_ai == current_player:
            for _ in range(search_sims):
                mcts.search(s, game, net, c_puct)
            pi = mcts.pi(s)
            move = np.random.choice(len(pi), p=pi)
            s = game.next_state(s, move)
        if game.game_ended(s):
            return game.game_rewards(s)
        if current_player == "X":
            current_player = "O"
        else:
            current_player = "X"

def play_rand(net, search_sims, c_puct):
    i = np.random.choice([0,1])
    p_ai = ["X", "O"][i]
    player = ["X", "O"][1-i]

    game = Game()
    s = game.get_initial_state(player=p_ai)

    current_player = "X"

    mcts = MCTS()

    while True:
        if player == current_player:
            move = players.random_player(s)
            s = game.next_state(s, move)
        elif p_ai == current_player:
            for _ in range(search_sims):
                mcts.search(s, game, net, c_puct)
            pi = mcts.pi(s)
            move = np.random.choice(len(pi), p=pi)
            s = game.next_state(s, move)
        if game.game_ended(s):
            return game.game_rewards(s)
        if current_player == "X":
            current_player = "O"
        else:
            current_player = "X"


def eval(search_sims, c_puct, eval_games, eta, lmbda, net=None):
    if net is None:
        net = Network([9,10,10,10,10], load=True)
    vsOP = []
    # vsRand = []
    for _ in range(eval_games):
        vsOP.append(play_OP(net, search_sims, c_puct))
        # vsRand.append(play_rand(net, search_sims, c_puct))
    print(f"Wins: {vsOP.count(1)}, Draws: {vsOP.count(0)}, Losses: {vsOP.count(-1)} vsOP, eta: {eta}, lmbda: {lmbda}, c_puct: {c_puct}")
    # print(f"Wins: {vsRand.count(1)}, Draws: {vsRand.count(0)}, Losses: {vsRand.count(-1)} vsRand, eta: {eta}, lmbda: {lmbda}, c_puct: {c_puct}")
    return [vsOP.count(1), vsOP.count(0), vsOP.count(-1), eta, lmbda, c_puct]



if __name__ == '__main__':
    eval()