from nnet import Network
from game import Game
from mcts import MCTS
import eval
import training

def main():
    # hyperparameters
    num_iterations = 10
    num_episodes = 100   # episodes of self-play per iteration
    search_sims = 100    # simulations per move for mcts
    eta = 0.0001        # gradient descent step size parameter
    lmbda = 0.001        # weights regularisation parameter
    c_puct = 0.1      # tree search exploration parameter
    net = Network([9,20,20,20,10])     # sizes controls the number of neurons in each layer of the network
    game = Game()
    net = training.policy_iteration(game, net, num_iterations, num_episodes, search_sims, eta, lmbda, c_puct)
    eval.main()

if __name__ == '__main__':
    main()
