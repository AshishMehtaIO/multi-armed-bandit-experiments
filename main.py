import bandit_environments
import bandit_agents
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_experiments(experiments, results):

    if not results:
        raise NotImplementedError
    sns.set()
    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Avg Reward')
    for r in range(len(results)):
        plt.plot(results[r][0], label=experiments[r])
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    for r in range(len(results)):
        plt.plot(results[r][1], label=experiments[r])
    plt.legend()
    plt.show()


def run_experiments(agent, env, num_exps=100, num_steps=10000, k=10, epsilon=0.01, alpha=0.1):
    avg_reward = np.zeros(num_steps, dtype=np.float64)
    percent_optimal_act = np.zeros(num_steps, dtype=np.float64)
    for exp in range(num_exps):
        A = agent(k, epsilon, alpha)
        E = env(k)

        for step in range(num_steps):
            act = A.sample_action()
            reward = E.step(act)
            ideal_avg_reward, ideal_action = E.ideal_arm()
            A.update_value(reward, act)
            avg_reward[step] = avg_reward[step] + (1 / (exp + 1)) * (reward - avg_reward[step])
            percent_optimal_act[step] = percent_optimal_act[step] + \
                                        (1 / (exp + 1)) * (float(ideal_action == act) - percent_optimal_act[step])
        # print('Average reward=%f, Ideal Avg=%f, Action taken=%d, Ideal action=%d' % (avg_reward,
        #                                                                              ideal_avg_reward,
        #                                                                              act,
        #                                                                              ideal_action))

    return [avg_reward, percent_optimal_act]


if __name__ == '__main__':
    results = []

    exps = ['e=0.1', 'e=0.01', 'greedy (e=0)']
    results.append(
        run_experiments(agent=bandit_agents.EGreedySampleAvgRewAgent, env=bandit_environments.StationaryKbandit,
                        num_exps=2000, num_steps=1000, k=10, epsilon=0.1))
    results.append(
        run_experiments(agent=bandit_agents.EGreedySampleAvgRewAgent, env=bandit_environments.StationaryKbandit,
                        num_exps=2000, num_steps=1000, k=10, epsilon=0.01))
    results.append(
        run_experiments(agent=bandit_agents.EGreedySampleAvgRewAgent, env=bandit_environments.StationaryKbandit,
                        num_exps=2000, num_steps=1000, k=10, epsilon=0.0))

    plot_experiments(exps, results)
