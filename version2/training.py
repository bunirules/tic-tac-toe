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
    player = np.random.choice([1, -1])
    s = game.get_initial_state(player)
    mcts = MCTS()
    cur_player = 1
    while True:
        for _ in range(search_sims):
            mcts.search(s, game, net, c_puct)
        pi = mcts.pi(s)
        examples.append([tuple(s), [pi, cur_player]])
        a = np.random.choice(len(pi), p=pi)
        s = game.next_state(s, a)
        cur_player *= -1
        if game.game_ended(s):
            examples.append([tuple(s), [pi, cur_player]])
            # add game outcome to pi vector
            r = game.game_rewards(s, cur_player)
            examples = assign_rewards(examples, r, cur_player)
            return examples

def assign_rewards(examples, reward, cur_player):
    if reward == 0:
        for i in range(len(examples)):
            examples[i][1][1] = 0
    else:
        for i, example in enumerate(examples):
            new = reward * ((-1) ** (example[1][1] != cur_player))
            examples[i][1][1] = new
    for i, example in enumerate(examples):
        new = np.array(example[0])
        examples[i][0] = new
    return examples