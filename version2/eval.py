from nnet import Network
from game import Game
from mcts import MCTS
import players
import numpy as np

def play_OP(net, search_sims, c_puct, game_num):
    choice = np.random.choice(range(100))
    show = False
    if choice == 4:
        show = True
    i = np.random.choice([-1,1])
    p_ai =  i
    player =  -i

    game = Game()
    s = game.get_initial_state(player=p_ai)
    if show:
        print(f"Showing Game number {game_num}, ai is player {p_ai}")
        print(s)
    current_player = 1

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
        current_player *= -1
        if show:
            print(s)
        if game.game_ended(s):
            return game.game_rewards(s, current_player)

def play_rand(net, search_sims, c_puct):
    i = np.random.choice([-1,1])
    p_ai = i
    player = -i

    game = Game()
    s = game.get_initial_state(player=p_ai)

    current_player = 1

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
        current_player *= -1


def eval(search_sims, c_puct, eval_games, eta, lmbda, net=None):
    if net is None:
        net = Network([9,10,10,10,10,10], load=True)
    vsOP = []
    # vsRand = []
    for i in range(eval_games):
        vsOP.append(play_OP(net, search_sims, c_puct, i))
        # vsRand.append(play_rand(net, search_sims, c_puct))
    print(f"Wins: {vsOP.count(1)}, Draws: {vsOP.count(0)}, Losses: {vsOP.count(-1)} vsOP, eta: {eta}, lmbda: {lmbda}, c_puct: {c_puct}")
    # print(f"Wins: {vsRand.count(1)}, Draws: {vsRand.count(0)}, Losses: {vsRand.count(-1)} vsRand, eta: {eta}, lmbda: {lmbda}, c_puct: {c_puct}")
    return [vsOP.count(1), vsOP.count(0), vsOP.count(-1), eta, lmbda, c_puct]



if __name__ == '__main__':
    eval()