import numpy as np
import pandas as pd
from game import Game


def f(x):
    return 1 / (1 + np.exp(-x))


def game_to_board(game):
    board = {"top left": " ", "top mid": " ", "top right": " ",
             "mid left": " ", "mid mid": " ", "mid right": " ",
             "bot left": " ", "bot mid": " ", "bot right": " "}
    for i, move in enumerate(game):
        if i % 2 == 0:
            board[move] = "X"
        elif i % 2 == 1:
            board[move] = "O"
    return board


class Player:
    def __init__(self):
        self.player = "X"
        self.moves = ["top left", "top mid", "top right",
                      "mid left", "mid mid", "mid right",
                      "bot left", "bot mid", "bot right"]
        self.possible_moves = self.moves
        self.game_history = pd.read_csv("game_history.csv")
        self.relevant_games = np.array(self.game_history)
        self.lost_games = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.current_game = []
        self.current_board = game_to_board(self.current_game)

    def check_relevant_game(self, game):
        game = game[1:len(self.current_game) + 1]
        game_board = game_to_board(game)
        if game_board == self.current_board:
            return True
        else:
            return False

    def update_relevant_games(self):
        ind_list = []
        for j, game in enumerate(self.relevant_games):
            if not self.check_relevant_game(game):
                self.lost_games = np.append(self.lost_games, [game], axis=0)
                ind_list.append(j)
        self.relevant_games = np.delete(self.relevant_games, ind_list, axis=0)
        # print(len(self.lost_games))
        if len(self.current_game) >= 5:
            k_list = []
            for k, game in enumerate(self.lost_games):
                if self.check_relevant_game(game):
                    self.relevant_games = np.append(self.relevant_games, [game], axis=0)
                    k_list.append(k)
            self.lost_games = np.delete(self.lost_games, k_list, axis=0)
        # print(len(self.lost_games))

    def get_previous_moves(self):
        n = len(self.current_game)
        move_info = dict([move, 0] for move in self.possible_moves)
        if len(self.relevant_games) > 0:
            move_list = self.relevant_games[:, [0,n+1,-1]]
            for move in move_list:
                # if the player made the move and won or if the opponent made the move and beat the player
                if move[0] == self.player == move[-1] or move[0] != self.player != move[-1]:
                    if move[-1] in ["X", "O"]:
                        # print(move_info)
                        # print(move)
                        # print(len(self.current_game))
                        move_info[move[1]] += 1
                # if the player made the move and lost or the opponent made the move and lost
                elif move[0] == self.player != move[-1] or move[0] != self.player == move[-1]:
                    if move[-1] in ["X", "O"]:
                        # print(move_info)
                        # print(move)
                        # print(len(self.current_game))
                        move_info[move[1]] -= 1

        tot = sum(move_info.values())
        for key in move_info:
            move_info[key] -= tot / len(move_info)
            move_info[key] = f(move_info[key])
        return move_info

    def analyse(self, game):
        self.current_game = game.current_game
        self.current_board = game_to_board(self.current_game)
        for move in self.current_game:
            if move in self.possible_moves:
                self.possible_moves.remove(move)
        if len(self.current_game) > 0:
            self.update_relevant_games()
        move_info = self.get_previous_moves()
        move_prob = [move_info[key] for key in move_info]
        tot = sum(move_prob)
        num = np.random.rand()
        for i in range(len(move_prob)):
            move_prob[i] /= tot
            if i >= 1:
                move_prob[i] += move_prob[i - 1]
            if num <= move_prob[i]:
                num = i
                break
        next_move = self.possible_moves[num]
        return next_move

    def add_game_to_history(self, game):
        self.current_game = game.current_game
        if len(self.current_game) < 9:
            for _ in range(9 - len(self.current_game)):
                self.current_game.append(None)
        self.current_game.insert(0, self.player)
        if game.draw:
            self.current_game.append("-")
        else:
            self.current_game.append(game.win)
        n = len(self.game_history)
        self.game_history.loc[n] = self.current_game
        self.relevant_games = np.array(self.game_history)
        df = self.game_history.iloc[[n]]
        df.to_csv("game_history.csv", mode="a", index=False, header=False)
        self.current_game = []
        self.possible_moves = self.moves
