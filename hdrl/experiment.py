import numpy as np

from hdrl.agents import Agent


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
    def __init__(self, agentargs, environment, trials, tmax, savedir, verbose=True, graphs=False, seed=0):
        """
        Object to house, manage, and execute experiments
            ARGS:
                agentargs(dict):= dictionary of keyword arguments for the Agent class
                environment:= environment object for the agent to interact with
                trials(int):= number of trials to evaluate each agent for
                tmax(int):= maximum number of time steps to train each agent for
                savedir(str):= location to save the results of the experiment
                verbose(bool):= True prints trial statistics and periodic training statistics
                graphs(bool):= create and save plots of the returns for each trial
                seed(int):= seed for reproducibility
        """
        self.agent_args = agentargs
        self.env = environment
        self.trials = trials
        self.t_max = tmax
        self.dir = savedir
        self.verbose = verbose
        self.graphs = graphs

        self.meta_data = {}
        self.results = {}

        self.seed = random.seed(seed)

        #Creates an array with a random seed for each trial
        self.seeds = np.random.randint(0, int(1e5), size=trials)

    def init_trial(self, seed):
        """
        Initializes an agent and results dictionary for a trial
        """
        self.agent = Agent(**self.agent_args)

        trial_results = {}

        return trial_results

    def run_trial(self):
        pass

    def write_result(self):
        pass

    def run_experiments(self):
        pass

    def save_weights_and_results(self):
        pass

    def make_graph(self):
        pass

    def compute_stats_all_trials(self):
        pass
