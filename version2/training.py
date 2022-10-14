import numpy as np
from mcts import MCTS

def policy_iteration(game, net, num_iterations, num_episodes, search_sims, eta, lmbda, c_puct):
    examples = []
    # n = 0
    for i in range(num_iterations):
        for e in range(num_episodes):
            examples.append(execute_episode(game, net, search_sims, c_puct))
            print(f"episode {e} completed")
        # n += len(examples)
        net.train_new_data(examples, eta, lmbda) # n
        print(f"iteration {i} completed")
    net.save_network_params()
    return net


def execute_episode(game, net, search_sims, c_puct):
    examples = []
    player = np.random.choice(["X", "O"])
    s = game.get_initial_state(player)
    mcts = MCTS()

    while True:
        for i in range(search_sims):
            mcts.search(s, game, net, c_puct)
            # print(i)
            # print(mcts.visited)
            # print("-------")
            # print(s)
            # print("-------")
        pi = mcts.pi(s)
        examples.append([s, pi, None])
        a = np.random.choice(len(pi), p=pi)
        s = game.next_state(s, a)
        if game.game_ended(s):
            examples = assign_rewards(examples, game.game_rewards(s))
            return examples

def assign_rewards(examples, val):
    for example in examples:
        example[2] = val
    return examples