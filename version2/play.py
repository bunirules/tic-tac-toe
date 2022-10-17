from nnet import Network
from game import Game
from mcts import MCTS
import numpy as np

def print_board(s):
    bd = {1:"X", 0:" ", -1:"O"}
    print(f" {bd[s[0]]} | {bd[s[1]]} | {bd[s[2]]}")
    print("------------")
    print(f" {bd[s[3]]} | {bd[s[4]]} | {bd[s[5]]}")
    print("------------")
    print(f" {bd[s[6]]} | {bd[s[7]]} | {bd[s[8]]}")

def help():
    print("To make your move, enter a number 1-9")
    print("1 indicates the top left of the board, 2 indicates top middle etc.")
    print("9 indicates the bottom right of the board")

def play():
    ans = input("Do you want to be player X? (yes or no)")
    if ans == "yes":
        player = "X"
        p_ai = "O"
    elif ans == "no":
        player = "O"
        p_ai = "X"
    else:
        player = "X"
        p_ai = "O"
    print(f"You are player {player}")

    ai = Network([9,10,10,10,10])

    game = Game()
    s = game.get_initial_state(player=p_ai)

    c_puct = 0.1
    search_sims = 50

    current_player = "X"

    while True:
        if player == current_player:
            while True:
                try:
                    print_board(s)
                    help()
                    move = int(input("Your move: ")) - 1
                    s = game.next_state(s, move)
                    break
                except ValueError: 
                    pass
        elif p_ai == current_player:
            print_board(s)
            mcts = MCTS()
            for _ in range(search_sims):
                mcts.search(s, game, ai, c_puct)
            pi = mcts.pi(s)
            move = np.random.choice(len(pi), p=pi)
            s = game.next_state(s, move)
        if game.game_ended(s):
            print_board(s)
            print("Game over!")
            if game.game_rewards(s) == 0:
                print("Game is a draw.")
            elif game.game_rewards(s) == 1:
                print("The bot won this game. Unlucky!")
            elif game.game_rewards(s) == -1:
                print("You beat the bot. Nicely done!")
            break
        if current_player == "X":
            current_player = "O"
        else:
            current_player = "X"

if __name__ == '__main__':
    play()