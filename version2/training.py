import numpy as np
from mcts import MCTS

def policy_iteration(game, net, num_iterations, num_episodes, search_sims, eta, lmbda, c_puct):
    examples = []
    n = 0
    for i in range(num_iterations):
        for e in range(num_episodes):
            examples.append(execute_episode(game, net, search_sims, c_puct))
            n += len(examples[-1])
        for example in examples:
            net.train_new_data(example, eta, lmbda, n) # n
        print(f"iteration {i} completed")
    net.save_network_params()
    return net


def execute_episode(game, net, search_sims, c_puct):
    examples = []
    player = np.random.choice(["X", "O"])
    s = game.get_initial_state(player)
    mcts = MCTS()

    while True:
        for _ in range(search_sims):
            mcts.search(s, game, net, c_puct)
        pi = mcts.pi(s)
        examples.append([s, pi])
        a = np.random.choice(len(pi), p=pi)
        s = game.next_state(s, a)
        if game.game_ended(s):
            # add game outcome to pi vector
            examples = assign_rewards(examples, game.game_rewards(s))
            return examples

def assign_rewards(examples, val):
    for i, example in enumerate(examples):
        new = np.append(example[1], val)
        examples[i][1] = new
    return examples