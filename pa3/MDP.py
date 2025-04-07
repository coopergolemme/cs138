import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class MDP:
    def __init__(self, num_states, branching_factor, seed=None):
        self.num_states = num_states
        self.b = branching_factor
        self.terminal = -1
        self.transitions = {}
        self.rng = np.random.RandomState(seed)
        
        # Precompute transitions as NumPy arrays
        for s in range(num_states):
            self.transitions[s] = {}
            for a in [0, 1]:
                terminal_reward = self.rng.normal(0, 1)
                next_states = self.rng.choice(num_states, size=branching_factor, replace=True)
                next_rewards = self.rng.normal(0, 1, size=branching_factor)
                self.transitions[s][a] = (terminal_reward, next_states, next_rewards)

    def get_expected_reward_and_next_value(self, s, a, Q, epsilon, gamma=0.9):
        if s == self.terminal:
            return 0.0
        
        terminal_reward, next_states, next_rewards = self.transitions[s][a]
        expected_r = 0.1 * terminal_reward + 0.9 * np.mean(next_rewards)
        
        q0 = Q[next_states, 0]
        q1 = Q[next_states, 1]
        max_q = np.maximum(q0, q1)
        v = (1 - epsilon) * max_q + epsilon/2 * (q0 + q1)
        next_value = gamma * 0.9 * np.mean(v)
        
        return expected_r + next_value
    def is_terminal(self, s):
        return s == self.terminal

    def sample_next_state(self, s, a):
        _, next_states, _ = self.transitions[s][a]
        return np.random.choice(next_states)
def value_iteration(mdp, gamma=0.9, max_iter=1000):
    Q = np.zeros((mdp.num_states, 2))
    for _ in range(max_iter):
        new_Q = np.zeros_like(Q)
        for s in range(mdp.num_states):
            for a in [0, 1]:
                new_Q[s, a] = mdp.get_expected_reward_and_next_value(s, a, Q, 0, gamma)
        if np.max(np.abs(new_Q - Q)) < 1e-4:
            break
        Q = new_Q
    return Q

def run_experiment(seed, num_iterations=200_000, num_episodes=200_000, branching_factor=3):
    print(f"Starting task {seed}")
    # Create MDP instance
    mdp = MDP(num_states=10000, branching_factor=branching_factor, seed=seed)
    
    # Approximate optimal Q (not fully converged for speed)
    optimal_Q = value_iteration(mdp, gamma=0.9)
    optimal_start_value = np.max(optimal_Q[0])
    
    # Uniform updates
    Q_uni = np.zeros((mdp.num_states, 2))
    uni_values = []
    for _ in range(num_iterations):
        s = np.random.randint(10000)
        a = np.random.randint(2)
        Q_uni[s, a] = mdp.get_expected_reward_and_next_value(s, a, Q_uni, epsilon=0.1)
        uni_values.append(np.max(Q_uni[0]))
    
    # On-policy updates
    Q_on = np.zeros((mdp.num_states, 2))
    on_values = []
    for _ in range(num_episodes):
        s = 0
        episode = []
        while s != mdp.terminal:
            if np.random.rand() < 0.1:
                a = np.random.randint(2)
            else:
                a = np.argmax(Q_on[s])
            episode.append((s, a))
            
            if np.random.rand() < 0.1:
                s = mdp.terminal
            else:
                s = mdp.transitions[s][a][1][0]  # b=1 direct transition
            
        for (s, a) in episode:
            Q_on[s, a] = mdp.get_expected_reward_and_next_value(s, a, Q_on, epsilon=0.1)
        on_values.append(np.max(Q_on[0]))
    

    return np.array(uni_values), np.array(on_values), optimal_start_value

# Add this at the very end of your script (after all function/class definitions)
if __name__ == "__main__":
    # Experiment parameters
    num_tasks = 1
    num_iterations = 10_000
    num_episodes = 10_000
    epsilon = 0.1
    branching_factor = 3

    # Parallel execution
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, seed, num_iterations, num_episodes, branching_factor) for seed in range(num_tasks)]
        results = [f.result() for f in tqdm(futures, desc="Processing MDPs")]
    # Rest of your aggregation and plotting code
    all_uni = np.array([r[0] for r in results])
    all_on = np.array([r[1] for r in results])
    all_mcts = np.array([r[2] for r in results])
    all_opt = np.array([r[3] for r in results])

    # Compute statistics
    uni_mean = np.mean(all_uni, axis=0)
    uni_std = np.std(all_uni, axis=0)
    on_mean = np.mean(all_on, axis=0)
    on_std = np.std(all_on, axis=0)
    opt_mean = np.mean(all_opt)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(uni_mean, label='Uniform Updates (b=3)', color='blue')
    plt.fill_between(range(num_iterations), uni_mean - uni_std, uni_mean + uni_std, alpha=0.2, color='blue')
    plt.plot(on_mean, label='On-policy Updates (b=3)', color='orange')
    plt.fill_between(range(num_episodes), on_mean - on_std, on_mean + on_std, alpha=0.2, color='orange')
    plt.axhline(opt_mean, color='red', linestyle='--', label='Average Optimal Value')
    plt.xlabel('Computation Time (Expected Updates)')
    plt.ylabel('Value of Start State')
    plt.title(f'10,000 States | Averaged Over {num_tasks} Tasks')
    plt.xlim(0, 200000)
    plt.legend()
    plt.grid(True)
    plt.show()
