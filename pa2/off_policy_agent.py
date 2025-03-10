import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from env import RaceTrack

class OffPolicyMCAgent:
    def __init__(self,
                env: RaceTrack,
                epsilon=0.3,
                gamma=0.9,
                reward_finish=0,
                reward_crash=-1,
                reward_time_step=-1):
        
        self.env = env
        self.epsilon = epsilon  # Soft policy exploration rate
        self.gamma = gamma  # Discount factor
        self.reward_finish = reward_finish
        self.reward_crash = reward_crash
        self.reward_time_step = reward_time_step
        
        self.action_space = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        self.bad_starting_actions = [(0, 0), (-1, 0), (0, 1), (-1, 1)]
        self.Q_s_a = defaultdict(float)  
        self.C_s_a = defaultdict(float) 
        self.policy = {}  # Target policy
        random.seed(None);
        
        self.__initialize_policy()
    
    def __initialize_policy(self):
        for x in range(self.env.width):
            for y in range(self.env.height):
                for vx in range(5):
                    for vy in range(5):
                        if ((x, y) in self.env.start_line):
                            self.policy[((x, y), (vx, vy))] = random.choice([a for a in self.action_space if a not in self.bad_starting_actions])
                        else:
                            self.policy[((x, y), (vx, vy))] = random.choice(self.action_space)
    
    def behavior_policy(self, state, velocity):
        """Returns an action following a soft policy (ε-soft or random)."""
        state_key = (state, velocity)
        if state_key not in self.policy:
            self.policy[state_key] = random.choice(self.action_space)  # Initialize missing states
        
        if random.random() < self.epsilon:
            valid_actions = [a for a in self.action_space if a not in self.bad_starting_actions] if state in self.env.start_line else self.action_space
            return random.choice(valid_actions)
        else:
            return self.policy[state_key]
    
    
    def step(self, state, velocity, action):
        """Performs an environment step, returning (next_state, new_velocity, reward, done)."""
        new_velocity = (min(max(velocity[0] + action[0], 0), 4),
                        max(min(velocity[1] + action[1], 0), -4))
        new_position = (state[0] + new_velocity[0], state[1] + new_velocity[1])
        
        if self.env.check_crash(state, new_velocity):
            new_position = random.choice(self.env.start_line)  # Reset position
            new_velocity = (0, 0)  # Reset velocity
            return new_position, new_velocity, self.reward_crash, False
        
        if new_position in self.env.finish_line:
            return new_position, new_velocity, self.reward_finish, True  # Reached finish line
        
        if new_velocity == (0,0) and new_position not in self.env.start_line:
            while new_velocity == (0,0):
                action = random.choice(self.action_space)
                new_velocity = (min(max(velocity[0] + action[0], 0), 4),
                                 max(min(velocity[1] + action[1], 0), -4))
        return new_position, new_velocity, self.reward_time_step, False  # Regular move
    
    def generate_episode(self):
        """Generates an episode using the behavior policy."""
        random.seed(None);
        state = random.choice(self.env.start_line)
        velocity = (0, 0)
        episode = []
        done = False
        
        while not done:
            action = self.behavior_policy(state, velocity)
            next_state, next_velocity, reward, done = self.step(state, velocity, action)
            episode.append((state, velocity, action, reward))
            state, velocity = next_state, next_velocity
        
        return episode
    
    def update_policy(self, episode):
        """Performs Off-Policy MC Control updates using Weighted Importance Sampling."""
        G = 0
        W = 1  # Importance weight
        
        for t in reversed(range(len(episode))):
            state, velocity, action, reward = episode[t]
            state_key = (state, velocity)
            
            G = self.gamma * G + reward
            self.C_s_a[(state_key, action)] += W
            
            # Q-value weighted update
            self.Q_s_a[(state_key, action)] += (W / (self.C_s_a[(state_key, action)] + 1e-6)) * (G - self.Q_s_a[(state_key, action)])
            
            # Update policy greedily
            self.policy[state_key] = max(self.action_space, key=lambda a: self.Q_s_a[(state_key, a)])
            
            # If action was not selected by learned policy, stop updating
            if action != self.policy[state_key]:
                break
            
            # Update importance sampling weight
            W *= 1 / (self.epsilon + (1 - self.epsilon) * (action == self.policy[state_key]))
            W = min(W, 100)
    
    def train(self, episodes=5000) -> list:
        """Trains the agent using Off-Policy MC Control."""
        print("Training started for Off-Policy MC Control...")
        print("Params: epsilon:", self.epsilon, "gamma:", self.gamma, "reward_finish:", self.reward_finish)
        print("Training in progress...")
        testing_data = []
        for i in range(episodes):
            episode = self.generate_episode()
            self.update_policy(episode)
            if (i+1) % 1000 == 0:
                testing_data.append(self.test(episodes=1000))
        print("Training complete!")
        return testing_data
    
    def test(self, episodes=100):
        """Tests the agent using the learned policy."""
        total_steps = 0
        for _ in range(episodes):
            state = random.choice(self.env.start_line)
            velocity = (0, 0)
            done = False
            steps = 0
            
            while not done:
                action = self.policy.get((state, velocity), random.choice(self.action_space))
                state, velocity, _, done = self.step(state, velocity, action)
                steps += 1
            
            total_steps += steps
        
        avg_steps = total_steps / episodes
        return avg_steps
    def display_episode(self, save_to_file=False, filename="episode_trail.png"):
        """Displays the trail of the agent moving through the environment."""
        random.seed(None)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.env.track, cmap="Greys", origin="upper")
        colors = ['ro', 'bo', 'go', 'yo', 'mo', 'co']  
        car_markers = [ax.plot([], [], colors[i % len(colors)], markersize=10, )[0] for i in range(len(self.env.start_line))]
        plt.legend()
        
        done = [False] * len(self.env.start_line)
        states = list(self.env.start_line)
        velocities = [(0, 0)] * len(self.env.start_line)
        
        trails = [[] for _ in range(len(self.env.start_line))]
        
        while not all(done):
            for i in range(len(self.env.start_line)):
                if not done[i]:
                    action = self.policy.get((states[i], velocities[i]), random.choice(self.action_space))  # Handle missing states
                    next_state, next_velocity, _, done[i] = self.step(states[i], velocities[i], action)
                    car_marker = car_markers[i]
                    car_marker.set_data([next_state[0]], [next_state[1]])
                    states[i], velocities[i] = next_state, next_velocity
                    trails[i].append(next_state)
        
        for i, trail in enumerate(trails):
            trail = np.array(trail)
            ax.plot(trail[:, 0], trail[:, 1], colors[i % len(colors)], )
        
        plt.title("Agent's Trail on the Track")
        
        if save_to_file:
            plt.savefig(filename)
            print(f"Trail saved to {filename}")
        else:
            plt.show()
