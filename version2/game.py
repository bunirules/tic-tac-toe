import numpy as np

class Game:
    def get_initial_state(self, player):
        self.player = player
        return np.array([0,0,0,0,0,0,0,0,0])

    @staticmethod
    def game_ended(s):
        board = s.reshape([3,3])
        for row in board:
            if abs(sum(row)) == 3:
                return True
        for col in board.T:
            if abs(sum(col)) == 3:
                return True
        if abs(s[0] + s[4] + s[8]) == 3:
            return True
        elif abs(s[2] + s[4] + s[6]) == 3:
            return True
        elif 0 not in s:
            return True
        else:
            return False

    def game_rewards(self, s):
        if not self.game_ended(s):
            print("Game not ended")
            raise ValueError
        out = None
        if self.player == "X":
            mult = 1
        elif self.player == "O":
            mult = -1
        board = s.reshape([3,3])
        for row in board:
            if sum(row) == 3:
                out = 1
        for col in board.T:
            if sum(col) == 3:
                out = 1
        if out == 1:
            out = 1
        elif s[0] + s[4] + s[8] == 3:
            out = 1
        elif s[2] + s[4] + s[6] == 3:
            out = 1
        elif 0 not in s:
            out = 0
        else:
            out = -1
        return out*mult

    @staticmethod
    def get_valid_actions(s):
        valid_a = []
        for i in range(len(s)):
            if s[i] == 0:
                valid_a.append(i)
        return valid_a

    @staticmethod
    def next_state(s, a):
        if np.sum(s) == 0 and s[a] == 0:
            s[a] += 1
        elif np.sum(s) == 1 and s[a] == 0:
            s[a] -= 1
        else:
            print(f"Error getting next state \n s = {s} \n a = {a}")
            raise ValueError
        return s


