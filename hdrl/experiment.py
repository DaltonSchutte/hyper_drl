import os
import datetime.datetime as dt
import csv

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import gym

from hdrl.agents import Agent

"""
NOTES:
    Add evaluation method that runs the trained agent for n instances and saves results
"""


class Experiment:
    """
    Experiment class that runs a specified number of trials.
    A single trial is a single execution of:
        Random initialization of the agent and environment
        Allow the agent to interact with the environment for the maximum
            number of time steps with environment resets as the agent
            reaches the terminal state
        Save array of returns
        Generate and save any graphs

    It is known that deep reinforcement learning is sensitive to the random
        seed and it is recommended to set the number of trials to at least 3
    """
    def __init__(self, agentargs, envname, trials, epmax, savedir, verbose=True, graphs=False, seed=0):
        """
        Object to house, manage, and execute experiments
            ARGS:
                agentargs(dict):= dictionary of keyword arguments for the Agent class
                envname(str):= atari gym environment name
                trials(int):= number of trials to evaluate each agent for
                epmax(int):= maximum number of episodes to train each agent for
                savedir(str):= location to save the results of the experiment
                verbose(bool):= True prints trial statistics and periodic training statistics
                graphs(bool):= create and save plots of the returns for each trial
                seed(int):= seed for reproducibility
        """
        self.agent_args = agentargs
        self.env_name = envname
        self.trials = trials
        self.ep_max = epmax
        self.dir = savedir
        self.verbose = verbose
        self.graphs = graphs
        self.seed = random.seed(seed)

        # Creates an array with a random seed for each trial
        self.seeds = np.random.randint(0, int(1e5), size=trials)

        self.meta_data = {'Seeds': self.seeds,
                          'Agents': self.agent_args,
                          'Env': self.env_name,
                          'No. Trials': self.trials,
                          'Max Episodes': self.ep_max,
                          'Date': dt.now()
                          }
        self.results = {}

    def init_trial(self, seed):
        """
        Initializes an agent and results dictionary for a trial
        """
        env = gym.make(self.env_name)
        env.reset()

        self.agent_args.update({'state_size': env.observation_space.shape,  # Not compatible with mig=True, will fix
                                'action_size': env.action_space.n
                                })

        agent = Agent(**self.agent_args)
        agent.init_components(seed)

        trial_results = []

        return agent, env, trial_results

    def run_trial(self, trial_num, seed):
        """
        Runs the trial for the maximum number of time steps
        """
        agent, env, results = self.init_trial(seed)

        eps = self.epsilon

        for _ in range(self.ep_max):
            score = 0
            state = env.reset()
            while True:
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                score += reward
                state = next_state

                if done:
                    break

            results.append(score)

        self.save_weights_and_results(trial_num, agent, results)

        if self.verbose:
            write_result(trial_num, results)

        if self.graphs:
            self.make_graphs(trial_num, results)

        return results

    def run_experiments(self):
        pass

    def save_weights_and_results(self, trial_num, agent, results):
        """
        Saves the weights of the models in each agent
        """
        path = os.join(self.dir, f"trial_{trial_num}")
        os.mkdir(os.join(path))

        # Save model weights
        torch.save(agent.qnet_online.state_dict(), os.join(path, "online_weights.pth"))
        torch.save(agent.qnet_target.state_dict(), os.join(path, "target_weights.pth"))

        # Save results as csv
        with open(os.join(path, "results.csv"), 'w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerows(results)
            f.close()

    def make_graphs(self, trial_num, results):
        """
        Saves a plot of the reward for each episode and the moving average over 100 episodes
        """
        sns.set_style('darkgrid')

        path = os.join(self.dir, f"trial_{trial_num}")

        try:
            os.mkdir(os.join(path))
        except:
            pass

        # Save plot of the rewards
        results_array = np.array(results)
        n = len(results_array)
        ax = np.arance(0,n)

        plt.plot(ax, results_array)
        plt.savefig(os.join(path, "rewards.pdf"), dpi=1600)

        # Save plot of the moving average of the rewards
        moving_avg = np.empty(n)
        for i in range(n):
            moving_avg[i] = results_array[max(0, i-100):i+1].mean()

        plt.plot(ax, running_avg)
        plt.savefig(os.join(path, "rewards_ma.pdf"), dpi=1600)

    def compute_stats_all_trials(self):
        pass


# noinspection PyArgumentList
def write_result(trial_num, results):
    """
    Prints statistics for the trial
    """
    results_array = np.array(results)
    mean = results_array.mean()
    std = results_array.std()
    max_reward = results_array.max()
    print(f"===\t===\tTrial {trial_num}\t===\t===")
    print("\tMean {:.4f}\n\tStdDev: {:.4f}\n\tMax: {:.4f}\n".format(mean, std, max_reward))
