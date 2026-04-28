from pond_rl.agents.base_agent import BaseAgent
from pond_rl.agents.random_agent import RandomAgent
from pond_rl.agents.dqn import DQNAgent
from pond_rl.agents.dqn_target import DQNTargetAgent
from pond_rl.agents.dqn_experience_replay import DQNExperienceReplayAgent
from pond_rl.agents.dqn_prioritized_replay import DQNPrioritizedReplayAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "dqn": DQNAgent,
    "dqn_target": DQNTargetAgent,
    "dqn_er": DQNExperienceReplayAgent,
    "dqn_per": DQNPrioritizedReplayAgent,
}

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "DQNAgent",
    "DQNTargetAgent",
    "DQNExperienceReplayAgent",
    "DQNPrioritizedReplayAgent",
    "AGENT_REGISTRY",
]
