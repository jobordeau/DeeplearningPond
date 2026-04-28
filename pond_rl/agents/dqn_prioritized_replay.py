import os

import numpy as np
import torch

from pond_rl.agents.dqn_target import DQNTargetAgent
from pond_rl.utils.evaluation import evaluate_agent
from pond_rl.utils.model_io import save_model, update_best_models
from pond_rl.utils.prioritized_replay_buffer import PrioritizedReplayBuffer


class DQNPrioritizedReplayAgent(DQNTargetAgent):
    name = "dqn_per"

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
        buffer_capacity=10000,
        batch_size=64,
        warmup_steps=200,
        alpha=0.6,
        beta_start=0.4,
        beta_increment=1e-4,
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
            target_update_freq=target_update_freq,
            device=device,
        )
        self.buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha)
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.beta = beta_start
        self.beta_increment = beta_increment

    def _learn(self):
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None
        sample = self.buffer.sample(self.batch_size, beta=self.beta)
        states, actions, rewards, next_states, next_masks, dones, weights, indices = sample

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_masks = next_masks.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values_masked = next_q_values.clone()
            next_q_values_masked[~next_masks] = -float("inf")
            no_valid = ~next_masks.any(dim=1)
            next_q_values_masked[no_valid] = 0.0
            max_next_q = next_q_values_masked.max(dim=1)[0]
            max_next_q[no_valid] = 0.0
            targets = rewards + self.gamma * (1 - dones) * max_next_q

        predicted = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = (predicted - targets).detach().cpu().numpy()
        loss = (weights * (predicted - targets) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.update_priorities(indices, np.abs(td_errors))
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
                self.buffer.store(state, action_idx, reward, next_state, next_mask, done)
                self._learn()

            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_network()

            self.decay_epsilon()
            self.beta = min(1.0, self.beta + self.beta_increment)

            if (episode + 1) % eval_interval == 0:
                metrics = evaluate_agent(env, self, num_eval_episodes=eval_episodes)
                if verbose:
                    print(f"[{self.name}] Episode {episode + 1}/{num_episodes} | "
                          f"Win Rate: {metrics['win_rate']:.2f}% | "
                          f"Lose Rate: {metrics['lose_rate']:.2f}% | "
                          f"Tie Rate: {metrics['tie_rate']:.2f}% | "
                          f"Epsilon: {self.epsilon:.3f} | Beta: {self.beta:.3f}")
                model_path = save_model(self.q_network, metrics["win_rate"], folder=save_folder, prefix=self.name)
                best_models = update_best_models(metrics["win_rate"], model_path, best_models)

        if best_models and verbose:
            print(f"\nTop {len(best_models)} models for {self.name}:")
            for win_rate, path in best_models:
                print(f"  - {path} | Win Rate: {win_rate:.2f}%")
        return best_models
