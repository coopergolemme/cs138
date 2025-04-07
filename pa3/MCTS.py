import math
import random
import matplotlib.pyplot as plt


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent  # Parent node.
        self.action = action  # Action taken to reach this node.
        self.children = {}    # Mapping from action -> child node.
        self.visits = 0
        self.value = 0.0
        # Untried actions for this state.
        self.untried_actions = None


def generate_tree_mdp(depth, branching_factor, reward_range=(0, 10), seed=42):
    """
    Generates a tree-structured MDP.
    Each non-terminal state has 'branching_factor' actions.
    The reward for each transition is drawn uniformly from reward_range.
    Returns a dictionary mapping state IDs to lists of (action, next_state, reward) tuples,
    the root state ID, and the total number of states.
    """
    random.seed(seed)
    mdp = {}
    state_id = 0

    def add_node(current_depth):
        nonlocal state_id
        current_state = state_id
        state_id += 1
        if current_depth < depth:
            children = []
            for a in range(branching_factor):
                child = add_node(current_depth + 1)
                reward = random.uniform(*reward_range)
                children.append((a, child, reward))
            mdp[current_state] = children
        else:
            mdp[current_state] = []  # Terminal state.
        return current_state

    root = add_node(0)
    return mdp, root, state_id


def mcts(mdp, root_state, gamma, n_iter, C=1.0, rollout_depth=10):
    """
    Runs Monte Carlo Tree Search (MCTS) starting at the given root_state.
    Performs n_iter iterations (each consisting of selection, expansion, simulation, backup)
    and returns the best action from the root along with the final root node.
    """
    root = MCTSNode(root_state)
    root.untried_actions = [a for (a, _, _) in mdp[root_state]]
    
    for _ in range(n_iter):
        node = root
        state = root_state

        # --- Selection ---
        while node.untried_actions == [] and mdp[state]:
            ucb_values = {}
            for action, child in node.children.items():
                if child.visits == 0:
                    ucb_values[action] = float('inf')
                else:
                    ucb_values[action] = child.value/child.visits + C * math.sqrt(math.log(node.visits + 1) / child.visits)
            best_action = max(ucb_values, key=ucb_values.get)
            node = node.children[best_action]
            transition = next(item for item in mdp[state] if item[0] == best_action)
            state = transition[1]

        # --- Expansion ---
        if mdp[state] and node.untried_actions:
            a = random.choice(node.untried_actions)
            node.untried_actions.remove(a)
            transition = next(item for item in mdp[state] if item[0] == a)
            next_state = transition[1]
            r = transition[2]
            child_node = MCTSNode(next_state, parent=node, action=a)
            child_node.untried_actions = [x for (x, _, _) in mdp[next_state]] if mdp[next_state] else []
            node.children[a] = child_node
            node = child_node
            state = next_state
            total_reward = r
        else:
            total_reward = 0

        # --- Simulation (Rollout) ---
        current_depth = 0
        discount = 1.0
        while mdp[state] and current_depth < rollout_depth:
            action_tuple = random.choice(mdp[state])
            a, next_state, r = action_tuple
            total_reward += discount * r
            discount *= gamma
            state = next_state
            current_depth += 1

        # --- Backup ---
        backup_node = node
        while backup_node is not None:
            backup_node.visits += 1
            backup_node.value += total_reward
            total_reward *= gamma
            backup_node = backup_node.parent

    # Choose the best action from the root based on visit count.
    best_action = None
    best_visits = -1
    for a, child in root.children.items():
        if child.visits > best_visits:
            best_visits = child.visits
            best_action = a
    return best_action, root



def simulate_episode(mdp, start_state, gamma, n_iter, C, rollout_depth):
    """
    Simulates an episode in the MDP using MCTS for decision-time planning.
    At each decision point, MCTS is used to select an action.
    Returns the cumulative discounted reward for the episode.
    """
    state = start_state
    cumulative_reward = 0
    discount = 1.0

    while mdp[state]:  # while non-terminal
        best_action, _ = mcts(mdp, state, gamma, n_iter, C, rollout_depth)
        transition = next(item for item in mdp[state] if item[0] == best_action)
        a, next_state, r = transition
        cumulative_reward += discount * r
        discount *= gamma
        state = next_state
    return cumulative_reward

# |-----------------------------|
# |         Experiments         |
# |-----------------------------|
def experiment_varying_iterations(mdp, start_state, gamma, iterations_list, C, rollout_depth, episodes=50, num_runs=10):
    rewards = []
    for n_iter in iterations_list:
        run_rewards = []
        for run in range(num_runs):
            episode_rewards = []
            for _ in range(episodes):
                r = simulate_episode(mdp, start_state, gamma, n_iter, C, rollout_depth)
                episode_rewards.append(r)
            run_avg = sum(episode_rewards) / episodes
            run_rewards.append(run_avg)
        avg_reward = sum(run_rewards) / num_runs
        rewards.append(avg_reward)
    return rewards

def experiment_varying_rollout(mdp, start_state, gamma, n_iter, C, rollout_list, episodes=50, num_runs=10):
    rewards = []
    for rollout_depth in rollout_list:
        run_rewards = []
        for run in range(num_runs):
            episode_rewards = []
            for _ in range(episodes):
                r = simulate_episode(mdp, start_state, gamma, n_iter, C, rollout_depth)
                episode_rewards.append(r)
            run_avg = sum(episode_rewards) / episodes
            run_rewards.append(run_avg)
        avg_reward = sum(run_rewards) / num_runs
        rewards.append(avg_reward)
    return rewards

def experiment_varying_exploration(mdp, start_state, gamma, n_iter, rollout_depth, C_list, episodes=50, num_runs=10):
    rewards = []
    for C in C_list:
        run_rewards = []
        for run in range(num_runs):
            episode_rewards = []
            for _ in range(episodes):
                r = simulate_episode(mdp, start_state, gamma, n_iter, C, rollout_depth)
                episode_rewards.append(r)
            run_avg = sum(episode_rewards) / episodes
            run_rewards.append(run_avg)
        avg_reward = sum(run_rewards) / num_runs
        rewards.append(avg_reward)
    return rewards


if __name__ == '__main__':

    # MDP parameters
    depth = 5
    branching_factor = 10
    gamma = 0.9


    mdp, root_state, total_states = generate_tree_mdp(depth, branching_factor)


    print(f"Generated MDP with {total_states} states (depth {depth}, branching factor {branching_factor}).")
    
    # Experiment parameters.
    episodes = 100  # number of episodes per run for each parameter setting
    num_runs = 10  # number of independent runs to average over

    # Experiment 1: Varying number of MCTS iterations per decision.
    iterations_list = [1, 5, 10, 20, 30, 40]
    rewards_iterations = experiment_varying_iterations(mdp,
                                                        root_state,
                                                        gamma,
                                                        iterations_list,
                                                        C=0.5,
                                                        rollout_depth=5,
                                                        episodes=episodes,
                                                        num_runs=num_runs)

    # Experiment 2: Varying rollout depth.
    rollout_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rewards_rollout = experiment_varying_rollout(mdp, 
                                                 root_state,
                                                 gamma,
                                                 n_iter=20, 
                                                 C=0.5, 
                                                 rollout_list=rollout_list, 
                                                 episodes=episodes, 
                                                 num_runs=num_runs)

    # Experiment 3: Varying exploration constant (C).
    C_list = [0.1, 0.5, 0.75, 1.0, 1.5, 2.0] 
    rewards_exploration = experiment_varying_exploration(mdp,
                                                         root_state,
                                                         gamma, 
                                                         n_iter=20, 
                                                         rollout_depth=1, 
                                                         C_list=C_list, 
                                                         episodes=episodes, 
                                                         num_runs=num_runs)

    # Plot the results.
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(iterations_list, rewards_iterations, marker='o')
    plt.xlabel("MCTS Iterations per Decision")
    plt.ylabel("Average Episode Reward")
    plt.title("Varying MCTS Iterations")

    plt.subplot(1, 3, 2)
    plt.plot(rollout_list, rewards_rollout, marker='o')
    plt.xlabel("Rollout Depth")
    plt.ylabel("Average Episode Reward")
    plt.title("Varying Rollout Depth")

    plt.subplot(1, 3, 3)
    plt.plot(C_list, rewards_exploration, marker='o')
    plt.xlabel("Exploration Constant (C)")
    plt.ylabel("Average Episode Reward")
    plt.title("Varying Exploration Constant")

    plt.tight_layout()
    plt.show() 
