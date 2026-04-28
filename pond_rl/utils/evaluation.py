import time

import numpy as np


def evaluate_agent(env, agent, num_eval_episodes=200, measure_time=True):
    results = {"win": 0, "lose": 0, "tie": 0}
    episode_lengths = []
    move_times = []

    for _ in range(num_eval_episodes):
        env.reset()
        done = False
        steps = 0

        while not done:
            env.available_actions()
            if not np.any(env.action_mask) or env.board.game_over:
                break
            state = env.encode_state()
            action_mask = env.action_mask.copy()

            t0 = time.perf_counter() if measure_time else 0
            action_idx = agent.select_action(state, action_mask, greedy=True)
            if measure_time:
                move_times.append(time.perf_counter() - t0)

            _, _, done = env.step_with_action_id(action_idx, play_random_after_agent=True)
            steps += 1

        episode_lengths.append(steps)
        winner = env.board.winner
        if winner == "light":
            results["win"] += 1
        elif winner == "dark":
            results["lose"] += 1
        else:
            results["tie"] += 1

    total = sum(results.values())
    win_rate = results["win"] / total * 100 if total else 0.0
    lose_rate = results["lose"] / total * 100 if total else 0.0
    tie_rate = results["tie"] / total * 100 if total else 0.0
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    avg_move_time = float(np.mean(move_times)) if move_times else 0.0

    return {
        "win_rate": win_rate,
        "lose_rate": lose_rate,
        "tie_rate": tie_rate,
        "avg_length": avg_length,
        "avg_move_time": avg_move_time,
    }
