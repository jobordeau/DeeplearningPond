import argparse
import os
import sys

import torch

from pond_rl.agents import AGENT_REGISTRY
from pond_rl.env.pond_env import PondEnv
from pond_rl.utils.evaluation import evaluate_agent


def _build_agent(agent_name, env, device=None):
    if agent_name not in AGENT_REGISTRY:
        valid = ", ".join(AGENT_REGISTRY.keys())
        raise SystemExit(f"Unknown agent '{agent_name}'. Valid options: {valid}")

    agent_cls = AGENT_REGISTRY[agent_name]
    state_dim = len(env.encode_state())
    action_dim = env.num_actions()

    if agent_name == "random":
        return agent_cls(state_dim=state_dim, action_dim=action_dim)
    return agent_cls(state_dim=state_dim, action_dim=action_dim, device=device)


def _resolve_device(arg_device):
    if arg_device is None or arg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg_device)


def cmd_train(args):
    env = PondEnv()
    device = _resolve_device(args.device)
    agent = _build_agent(args.agent, env, device=device)
    save_folder = args.save_folder or os.path.join("models", args.agent)
    os.makedirs(save_folder, exist_ok=True)
    print(f"Training agent '{args.agent}' for {args.episodes} episodes on {device}...")
    agent.train(
        env,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_folder=save_folder,
    )


def cmd_evaluate(args):
    env = PondEnv()
    device = _resolve_device(args.device)
    agent = _build_agent(args.agent, env, device=device)
    if args.model:
        if not os.path.exists(args.model):
            raise SystemExit(f"Model file not found: {args.model}")
        agent.load(args.model)
    elif args.agent != "random":
        print("Warning: no model provided, evaluating an untrained agent.", file=sys.stderr)

    print(f"Evaluating agent '{args.agent}' over {args.episodes} episodes...")
    metrics = evaluate_agent(env, agent, num_eval_episodes=args.episodes)
    print("\n=== Evaluation results ===")
    print(f"Win Rate     : {metrics['win_rate']:.2f}%")
    print(f"Lose Rate    : {metrics['lose_rate']:.2f}%")
    print(f"Tie Rate     : {metrics['tie_rate']:.2f}%")
    print(f"Avg length   : {metrics['avg_length']:.2f} steps")
    print(f"Avg move time: {metrics['avg_move_time'] * 1000:.3f} ms")


def cmd_play(args):
    from pond_rl.gui.play import play_human_vs_agent, watch_agent_vs_random, watch_random_vs_random

    if args.mode == "human_vs_random":
        from pond_rl.agents import RandomAgent
        env = PondEnv()
        random_agent = RandomAgent(state_dim=len(env.encode_state()), action_dim=env.num_actions())
        play_human_vs_agent(random_agent, human_color=args.human_color)
        return

    if args.mode == "watch_random":
        watch_random_vs_random()
        return

    if args.mode in ("human_vs_agent", "watch_agent"):
        if args.agent is None:
            raise SystemExit("--agent is required for this play mode.")
        env = PondEnv()
        device = _resolve_device(args.device)
        agent = _build_agent(args.agent, env, device=device)
        if args.model:
            if not os.path.exists(args.model):
                raise SystemExit(f"Model file not found: {args.model}")
            agent.load(args.model)
        elif args.agent != "random":
            print("Warning: no model provided, agent is untrained.", file=sys.stderr)

        if args.mode == "human_vs_agent":
            play_human_vs_agent(agent, human_color=args.human_color)
        else:
            watch_agent_vs_random(agent, agent_color="light")
        return

    raise SystemExit(f"Unknown play mode: {args.mode}")


def cmd_benchmark(args):
    env = PondEnv()
    device = _resolve_device(args.device)
    rows = []

    print(f"Benchmark over {args.episodes} episodes per agent.\n")
    for agent_name in AGENT_REGISTRY.keys():
        agent = _build_agent(agent_name, env, device=device)
        model_path = None
        if args.models_dir:
            candidate_dir = os.path.join(args.models_dir, agent_name)
            if os.path.isdir(candidate_dir):
                pths = sorted(
                    [f for f in os.listdir(candidate_dir) if f.endswith(".pth")],
                    key=lambda s: float(s.rsplit("_", 1)[1].replace(".pth", "")) if "_" in s else 0.0,
                    reverse=True,
                )
                if pths:
                    model_path = os.path.join(candidate_dir, pths[0])

        if model_path and agent_name != "random":
            try:
                agent.load(model_path)
                model_label = os.path.basename(model_path)
            except Exception as exc:
                print(f"  ! could not load {model_path} for {agent_name}: {exc}")
                model_label = "no model"
        else:
            model_label = "no model" if agent_name != "random" else "n/a"

        metrics = evaluate_agent(env, agent, num_eval_episodes=args.episodes)
        rows.append((agent_name, model_label, metrics))

    header = f"{'Agent':<22} {'Model':<26} {'Win%':>7} {'Lose%':>7} {'Tie%':>7} {'AvgLen':>7} {'Move(ms)':>9}"
    print(header)
    print("-" * len(header))
    for name, label, m in rows:
        print(f"{name:<22} {label:<26} {m['win_rate']:>7.2f} {m['lose_rate']:>7.2f} "
              f"{m['tie_rate']:>7.2f} {m['avg_length']:>7.2f} {m['avg_move_time'] * 1000:>9.3f}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pond-rl",
        description="Reinforcement Learning agents for the Pond board game.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train an agent.")
    train.add_argument("--agent", required=True, choices=list(AGENT_REGISTRY.keys()))
    train.add_argument("--episodes", type=int, default=10000)
    train.add_argument("--eval-interval", type=int, default=200)
    train.add_argument("--eval-episodes", type=int, default=100)
    train.add_argument("--save-folder", type=str, default=None)
    train.add_argument("--device", type=str, default="auto")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="Evaluate a trained agent.")
    evaluate.add_argument("--agent", required=True, choices=list(AGENT_REGISTRY.keys()))
    evaluate.add_argument("--model", type=str, default=None)
    evaluate.add_argument("--episodes", type=int, default=1000)
    evaluate.add_argument("--device", type=str, default="auto")
    evaluate.set_defaults(func=cmd_evaluate)

    play = sub.add_parser("play", help="Launch the GUI.")
    play.add_argument(
        "--mode",
        type=str,
        default="human_vs_agent",
        choices=["human_vs_agent", "human_vs_random", "watch_agent", "watch_random"],
    )
    play.add_argument("--agent", type=str, default=None, choices=list(AGENT_REGISTRY.keys()))
    play.add_argument("--model", type=str, default=None)
    play.add_argument("--human-color", type=str, default="dark", choices=["light", "dark"])
    play.add_argument("--device", type=str, default="auto")
    play.set_defaults(func=cmd_play)

    benchmark = sub.add_parser("benchmark", help="Benchmark all agents.")
    benchmark.add_argument("--episodes", type=int, default=500)
    benchmark.add_argument("--models-dir", type=str, default="models")
    benchmark.add_argument("--device", type=str, default="auto")
    benchmark.set_defaults(func=cmd_benchmark)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
