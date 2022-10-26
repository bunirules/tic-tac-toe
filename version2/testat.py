from game import Game
from mcts import MCTS
from nnet import Network
import training
import numpy as np

def main():
    mcts = MCTS()
    game = Game()
    net = Network([9,10,10,10,10,10], load=True)
    examples = training.execute_episode(game, net, 50, 1)
    for i, example in enumerate(examples):
        print(f"{i+1}: {example}")
    # s = game.get_initial_state(player="X")
    # s = np.array([1,1,1,0,-1,0,-1,0,0])
    # p, v = net.predict(s)
    # print(p, v)
    # for i in range(10):
    #     mcts.search(s, game, net, 20)
    #     print(f"HAHAHAHAHAHHAHA: {i}")
    #     print("------")
    #     print("visited: ", mcts.visited)
    #     print("------")
    #     print("Q: ", mcts.Q)
    #     print("------")
    #     print("N: ",mcts.N)
    #     print("------")
    #     print("P: ",mcts.P)
    #     print("------")
    # print(np.random.choice([0,1,2,3,4,5,6,7,8,9],10, True, [1,0,0,0,0,0,0,0,0,0]))

if __name__ == '__main__':
    main()