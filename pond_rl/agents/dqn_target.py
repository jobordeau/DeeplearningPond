import os

import numpy as np
import torch
import torch.optim as optim

from pond_rl.agents.dqn import DQNAgent
from pond_rl.networks.q_network import QNetwork
from pond_rl.utils.evaluation import evaluate_agent
from pond_rl.utils.model_io import save_model, update_best_models


class DQNTargetAgent(DQNAgent):
    name = "dqn_target"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        target_update_freq=100,
        device=None,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            device=device,
        )
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.target_update_freq = target_update_freq

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_step(self, state, action_idx, reward, next_state, next_action_mask, done):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_q = float(reward)
            if not done and np.any(next_action_mask):
                mask_tensor = torch.tensor(next_action_mask, dtype=torch.bool, device=self.device)
                next_q_values = self.target_network(next_state_tensor)
                next_q_values_masked = next_q_values.clone()
                next_q_values_masked[~mask_tensor] = -float("inf")
                max_next = torch.max(next_q_values_masked).item()
                target_q += self.gamma * max_next

        predicted = self.q_network(state_tensor)[action_idx]
        target = torch.tensor(target_q, dtype=torch.float32, device=self.device)
        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, env, num_episodes=1000, eval_interval=100, eval_episodes=100, save_folder=None, verbose=True):
        save_folder = save_folder or os.path.join("models", self.name)
        best_models = []
        for episode in range(num_episodes):
            env.reset()
            done = False
            while not done:
                env.available_actions()
                if not np.any(env.action_mask) or env.board.game_over:
                    break
                state = env.encode_state()
                action_mask = env.action_mask.copy()
                action_idx = self.select_action(state, action_mask)
                next_state, reward, done = env.step_with_action_id(action_idx, play_random_after_agent=True)
                env.available_actions()
                next_mask = env.action_mask.copy()
                self._train_step(state, action_idx, reward, next_state, next_mask, done)

            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_network()

            self.decay_epsilon()

            if (episode + 1) % eval_interval == 0:
                metrics = evaluate_agent(env, self, num_eval_episodes=eval_episodes)
                if verbose:
                    print(f"[{self.name}] Episode {episode + 1}/{num_episodes} | "
                          f"Win Rate: {metrics['win_rate']:.2f}% | "
                          f"Lose Rate: {metrics['lose_rate']:.2f}% | "
                          f"Tie Rate: {metrics['tie_rate']:.2f}% | "
                          f"Epsilon: {self.epsilon:.3f}")
                model_path = save_model(self.q_network, metrics["win_rate"], folder=save_folder, prefix=self.name)
                best_models = update_best_models(metrics["win_rate"], model_path, best_models)

        if best_models and verbose:
            print(f"\nTop {len(best_models)} models for {self.name}:")
            for win_rate, path in best_models:
                print(f"  - {path} | Win Rate: {win_rate:.2f}%")
        return best_models
