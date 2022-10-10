import numpy as np


class Game:

    def __init__(self):
        self.current_player = 1
        self.win = None
        self.current_game = []
        self.board = np.array([[0,0,0],
                               [0,0,0],
                               [0,0,0]])
        self.win_combo = [[(0,0),(0,1),(0,2)],
                          [(1,0),(1,1),(1,2)],
                          [(2,0),(2,1),(2,2)],
                          [(0,0),(1,0),(2,0)],
                          [(0,1),(1,1),(2,1)],
                          [(0,2),(1,2),(2,2)],
                          [(0,0),(1,1),(2,2)],
                          [(2,0),(1,1),(0,2)]]
        self.symbols = [" ", "X", "O"]

    def print_board(self):
        print(f" {self.symbols[self.board[0,0]]} | {self.symbols[self.board[0,1]]} | {self.symbols[self.board[0,2]]}")
        print("------------")
        print(f" {self.symbols[self.board[1,0]]} | {self.symbols[self.board[1,1]]} | {self.symbols[self.board[1,2]]}")
        print("------------")
        print(f" {self.symbols[self.board[2,0]]} | {self.symbols[self.board[2,1]]} | {self.symbols[self.board[2,2]]}")

    def __print_message(self):
        print(f"It's your turn, player {self.current_player}!")

    def __check_win(self):
        for combo in self.win_combo:
            if (self.board[combo[0]] == self.current_player
                    and self.board[combo[1]] == self.current_player
                    and self.board[combo[2]] == self.current_player):
                self.win = self.current_player
                return True
        return False

    def __check_draw(self):
        for combo in self.win_combo:
            if (self.board[combo[0]] == 0
                    or self.board[combo[1]] == 0
                    or self.board[combo[2]] == 0
                    or self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]]):
                return False
        self.win = 0

        return True

    def __swap_player(self):
        if self.current_player == 1:
            self.current_player = -1
        else:
            self.current_player = 1

    def __player_won(self):
        self.print_board()
        print(f"Game won by player {self.symbols[self.current_player]}")

    def __game_draw(self):
        self.print_board()
        print("Game drawn")

    def player_move(self, space):
        if self.board[space] == 0:
            self.board[space] = self.current_player
            self.current_game.append(space)
            if self.__check_win():
                self.__player_won()
            elif self.__check_draw():
                self.__game_draw()
            else:
                self.__swap_player()
                self.print_board()
                self.__print_message()
            return True
        elif space != 0:
            print("Invalid move, try again")
            return False
