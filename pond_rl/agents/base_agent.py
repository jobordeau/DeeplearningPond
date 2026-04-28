from abc import ABC, abstractmethod


class BaseAgent(ABC):
    name = "base"

    @abstractmethod
    def select_action(self, state, action_mask, greedy=False):
        ...

    def train(self, env, num_episodes, eval_interval=100, eval_episodes=100, save_folder=None, verbose=True):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
