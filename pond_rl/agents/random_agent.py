import random

import numpy as np

from pond_rl.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, state_dim=None, action_dim=None, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, action_mask, greedy=False):
        valid_indices = np.flatnonzero(action_mask)
        if len(valid_indices) == 0:
            raise ValueError("No valid action available.")
        return int(random.choice(valid_indices))

    def train(self, env, num_episodes, eval_interval=100, eval_episodes=100, save_folder=None, verbose=True):
        from pond_rl.utils.evaluation import evaluate_agent
        if verbose:
            print(f"Random agent has nothing to train. Running {num_episodes} games...")
        metrics = evaluate_agent(env, self, num_eval_episodes=num_episodes)
        if verbose:
            print(f"Win Rate: {metrics['win_rate']:.2f}% | Lose Rate: {metrics['lose_rate']:.2f}% | "
                  f"Tie Rate: {metrics['tie_rate']:.2f}%")
        return metrics

    def save(self, path):
        pass

    def load(self, path):
        pass
