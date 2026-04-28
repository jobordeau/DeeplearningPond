import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def store(self, state, action, reward, next_state, next_action_mask, done):
        max_priority = self.priorities[: len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, next_action_mask, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, next_action_mask, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        size = len(self.buffer)
        priorities = self.priorities[:size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        weights = (size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, next_masks, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(next_masks), dtype=torch.bool),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) + 1e-6

    def __len__(self):
        return len(self.buffer)
