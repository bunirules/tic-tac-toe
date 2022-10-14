import numpy as np

class MCTS:
    def __init__(self):
        self.visited = []
        self.Q = []
        self.N = []
        self.P = []

    def search(self, s, game, net, c_puct):
        s = tuple(s)
        # print("1: ", s)
        if game.game_ended(np.array(s)):
            return -game.game_rewards(np.array(s))
        # print("2: ", s)
        if list(s) not in self.visited:
            self.visited.append(list(s))
            self.Q.append([0,0,0,0,0,0,0,0,0])
            self.N.append([0,0,0,0,0,0,0,0,0])
            new_p, v = net.predict(np.array(s))
            self.P.append(new_p)
            # print("8: ", s)
            return -v
        # print("3: ", s)
        i = self.visited.index(list(s))
        # print("4: ", s)
        u_max, best_a = -2, -2
        for a in game.get_valid_actions(s):
            u = self.Q[i][a] + c_puct*self.P[i][a]*(sum(self.N[i])/(1+self.N[i][a]))**0.5
            if u > u_max:
                u_max = u
                best_a = a
        a = best_a
        # print("5: ", s)
        sp = game.next_state(np.array(s), a)
        v = self.search(sp, game, net, c_puct)
        # print("6: ", s)
        self.Q[i][a] = (self.N[i][a]*self.Q[i][a] + v)/(self.N[i][a] + 1)
        self.N[i][a] += 1
        # print("7: ", s)
        return -v

    def pi(self, s):
        i = self.visited.index(list(s))
        vect = np.array(self.N[i])/sum(self.N[i])
        return vect


