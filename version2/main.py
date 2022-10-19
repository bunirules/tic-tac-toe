from nnet import Network
from game import Game
from mcts import MCTS
import eval
import training

def main():
    # hyperparameters
    num_iterations = 10
    num_episodes = 50   # episodes of self-play per iteration
    search_sims = 200    # simulations per move for mcts
    # etas = [0.0001,0.001, 0.01, 0.1]        # gradient descent step size parameter
    # lmbdas = [0.0001, 0.001, 0.01, 0.1]      # weights regularisation parameter
    # c_pucts = [0.1, 0.2, 0.3]      # tree search exploration parameter
    # eval_games = 100
    # eval_list = []
    # for eta in etas:
    #     eta_list = []
    #     for lmbda in lmbdas:
    #         lmbda_list = []
    #         for c_puct in c_pucts:
    #             net = Network([9,10,10,10,10])     # sizes controls the number of neurons in each layer of the network
    #             game = Game()
    #             net = training.policy_iteration(game, net, num_iterations, num_episodes, search_sims, eta, lmbda, c_puct)
    #             lmbda_list.append(eval.eval(search_sims, c_puct, eval_games, eta, lmbda))
    #         eta_list.append(lmbda_list)
    #     eval_list.append(eta_list)
    # print("eval_list: ", eval_list)

    eval_games = 1000
    eta = 0.0001
    lmbda = 0.1
    c_pucts = [3] #[1,2,3,4,5,6] # [0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    for c_puct in c_pucts:
        net = Network([9,10,10,10,10], load=True)     # sizes controls the number of neurons in each layer of the network
        game = Game()
        net = training.policy_iteration(game, net, num_iterations, num_episodes, search_sims, eta, lmbda, c_puct)

        
        print(eval.eval(search_sims, c_puct, eval_games, eta, lmbda))

if __name__ == '__main__':
    main()
