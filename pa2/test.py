from env import RaceTrack
from pa2.on_policy_agent import OnPolicyAgent
from off_policy_agent import OffPolicyMCAgent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def test_diff_alphas():
    track  = RaceTrack(seed=100)
    alphas = [0.01, 0.1, 0.2, 0.3]
    testing_data = []
    testing_trace_data = []
    for alpha in alphas:
        agent = OnPolicyAgent(track, epsilon=0.3, alpha=alpha, reward_finish=100);
        data = agent.train(episodes=50_000)
        testing_trace_data.append(data)

        testing_data.append(agent.test())
        # agent.display_episode()

    data = {
        'Agent': ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'],
        'Alpha': alphas,
        'Test Episodes': testing_data
    }

    df = pd.DataFrame(data)
    print(df)
    markers = ['o', 'x', '^', 's']
    for i, trace in enumerate(testing_trace_data):
        plt.plot(np.arange(1, len(trace) + 1) * 10, trace, label=f'Alpha {alphas[i]}', marker=markers[i])
    plt.xlabel('Training Steps (in tens of thousands)')
    plt.ylabel('Average Steps to Finish')
    plt.title('Training Performance for Different Alphas')
    plt.legend()
    plt.show()

def test_diff_gammas():
    track  = RaceTrack(seed=100)
    gammas = [0.85,0.9,0.95, 1.0]
    testing_data = []
    testing_trace_data = []
    for gamma in gammas:
        agent = OnPolicyAgent(track, epsilon=0.3, gamma=gamma,alpha=0.2, reward_finish=100);
        data = agent.train(episodes=70_000)
        testing_trace_data.append(data)

        testing_data.append(agent.test())
        # agent.display_episode()

    data = {
        'Agent': ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'],
        'Gamma': gammas,
        'Test Episodes': testing_data
    }

    df = pd.DataFrame(data)
    print(df)

    markers = ['o', 'x', '^', 's', 'D', 'P']
    for i, trace in enumerate(testing_trace_data):
        plt.plot(np.arange(1, len(trace) + 1) * 10, trace, label=f'Gamma {gammas[i]}', marker=markers[i])
    plt.xlabel('Training Steps (in tens of thousands)')
    plt.ylabel('Average Steps to Finish')
    plt.title('Training Performance for Different Gammas')
    plt.legend()
    plt.show()

def test_diff_finish_rewards():
    track  = RaceTrack(seed=100)
    finish_rewards = [75, 100, 200, 300]
    testing_data = []
    testing_trace_data = []
    for reward in finish_rewards:
        agent = OnPolicyAgent(track, epsilon=0.3, alpha=0.2, gamma=0.9, reward_finish=reward);
        data = agent.train(episodes=50_000)
        testing_trace_data.append(data)

        testing_data.append(agent.test())
        # agent.display_episode()

    data = {
        'Finish Reward': finish_rewards,
        'Test Episodes': testing_data
    }

    df = pd.DataFrame(data)
    print(df)

    markers = ['o', 'x', '^', 's', 'D']
    for i, trace in enumerate(testing_trace_data):
        plt.plot(np.arange(1, len(trace) + 1) * 1000, trace, label=f'Finish Reward {finish_rewards[i]}', marker=markers[i])
    plt.xlabel('Training Steps ')
    plt.ylabel('Average Steps to Finish')
    plt.title('Training Performance for Different Finish Rewards')
    plt.legend()
    plt.show()


def main():
    track = RaceTrack(seed=100)
    
    track.display_track()
    agent_1 = OnPolicyAgent(track, epsilon=0.3, alpha=0.01, reward_finish=100);
    agent_2 = OnPolicyAgent(track, epsilon=0.3, alpha=0.1, reward_finish=100);
    agent_3 = OnPolicyAgent(track, epsilon=0.3, alpha=0.2, reward_finish=100);
    agent_4 = OnPolicyAgent(track, epsilon=0.3, alpha=0.3, reward_finish=100);

    avg_ep_a1 = agent_1.train(episodes=50_000)
    avg_ep_a2 = agent_2.train(episodes=50_000)
    avg_ep_a3 = agent_3.train(episodes=50_000)
    avg_ep_a4 = agent_4.train(episodes=50_000)
    a1_test = agent_1.test()
    a2_test = agent_2.test()
    a3_test = agent_3.test()
    a4_test = agent_4.test()

    data = {
        'Agent': ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'],
        'Alpha': [0.01, 0.1, 0.2, 0.3],
        'Test Episodes': [a1_test, a2_test, a3_test, a4_test]
    }
    df = pd.DataFrame(data)
    print(df)
    
    val = input ("Enter model to test:")
    while(val != "exit"):
        switch = {
            '1': agent_1,
            '2': agent_2,
            '3': agent_3,
            '4': agent_4
            }
        agent = switch.get(val)
        agent.display_episode()
        val = input ("Enter model to test:")

def off_policy(num_episodes=100_000):
    track = RaceTrack(seed=100)
    agent = OffPolicyMCAgent(track, epsilon=0.3, gamma=0.9, reward_finish=200);
    avg_ep = agent.train(episodes=num_episodes);
    agent.display_episode(True, "off_policy_episode.png")
    return;

def on_policy(num_episodes=100_000):
    track = RaceTrack(seed=100)
    agent = OnPolicyAgent(track, epsilon=0.3, alpha=0.2, gamma=0.9, reward_finish=200);
    avg_ep = agent.train(episodes=num_episodes);
    agent.display_episode(True, "on_policy_episode.png")
    # return agent.test()

def compare_off_on():
    track = RaceTrack(seed=100)

    epsilon = 0.3
    alpha = 0.2
    gamma = 0.9
    reward_finish = 200
    num_episodes = 100_000


    
    agent_off = OffPolicyMCAgent(track, epsilon=epsilon, gamma=gamma, reward_finish=reward_finish);
    agent_on = OnPolicyAgent(track, epsilon=epsilon, alpha=alpha, gamma=gamma, reward_finish=reward_finish);

    off_training_data = agent_off.train(episodes=num_episodes)
    on_training_data = agent_on.train(episodes=num_episodes)

    plt.plot(np.arange(1, len(off_training_data) + 1) * 1000, off_training_data, label='Off-Policy', marker='o')
    plt.plot(np.arange(1, len(on_training_data) + 1) * 1000, on_training_data, label='On-Policy', marker='x')
    plt.xlabel('Episodes')
    plt.ylabel('Average Steps to Finish')
    plt.title('Off-Policy vs On-Policy Training')
    plt.legend()
    plt.show()

    agent_off.display_episode(save_to_file=True, filename="episode_animation_off.gif")
    agent_on.display_episode(save_to_file=True, filename="episode_animation_on.gif")


def smaller_off_policy(num_episodes=5000):
    track = RaceTrack(seed=102, width=100, height=50, track_width=10, move_up=0.65)
    track.display_track()
    agent = OffPolicyMCAgent(track, epsilon=0.3, gamma=0.9, reward_finish=10000)
    off_training_data = agent.train(episodes=num_episodes)

    plt.plot(off_training_data, label='Off-Policy', marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Average Steps to Finish')
    plt.title('Off-Policy Training on Smaller Track')
    plt.legend()
    plt.show()
    agent.display_episode()


    return agent.test()


def create_track():
    # track = RaceTrack()
    track2 = RaceTrack(seed=100)
    # track3 = RaceTrack()
    # track4 = RaceTrack()
    # track.display_track()

    track2.display_track()
    # track3.display_track()
    # track4.display_track()

def test_b():
    track = RaceTrack(seed=100)
    agent = OnPolicyAgent(track, epsilon=0.3, alpha=0.01, reward_finish=100);
    agent.display_episode()


if __name__ == '__main__':
    compare_off_on()
    # smaller_off_policy(500_000)
    # test_b()

    # off_policy()
    # on_policy();
    # create_track()
    # really_small_track()
    # main()
    # test_diff_alphas()
    # test_diff_gammas()
    # test_diff_finish_rewards()