import random
import time

import numpy as np
import pygame

from pond_rl.env.pond_env import PondEnv
from pond_rl.gui.display import COLORS, GameDisplay


def _init_pygame(caption):
    pygame.init()
    screen = pygame.display.set_mode((GameDisplay.WINDOW_WIDTH, GameDisplay.WINDOW_HEIGHT))
    pygame.display.set_caption(caption)
    font = pygame.font.SysFont(None, 32)
    return screen, font


def _agent_play_turn(env, agent):
    env.available_actions()
    if not np.any(env.action_mask):
        return False
    state = env.encode_state()
    mask = env.action_mask.copy()
    action_id = agent.select_action(state, mask, greedy=True)
    env.step_with_action_id(action_id, play_random_after_agent=False)
    return True


def _random_play_turn(env):
    actions = env.available_actions()
    if not actions:
        return False
    action = random.choice(actions)
    action_id = env.get_action_id(action)
    env.step_with_action_id(action_id, play_random_after_agent=False)
    return True


def play_human_vs_agent(agent, human_color="dark", ai_delay_ms=800):
    if human_color not in ("light", "dark"):
        raise ValueError("human_color must be 'light' or 'dark'.")
    agent_color = "light" if human_color == "dark" else "dark"

    screen, font = _init_pygame("Pond - Human vs Agent")
    clock = pygame.time.Clock()

    env = PondEnv()
    display = GameDisplay(screen, env.board, font, COLORS)
    display.set_status(f"You play {human_color}. Agent is {agent_color}.")

    last_ai_move_time = 0
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if env.board.turn == human_color and not env.board.game_over:
                    action_id = display.handle_human_click(env, human_color)
                    if action_id is not None:
                        env.step_with_action_id(action_id, play_random_after_agent=False)
                        last_ai_move_time = current_time

        if (
            not env.board.game_over
            and env.board.turn == agent_color
            and current_time - last_ai_move_time >= ai_delay_ms
        ):
            played = _agent_play_turn(env, agent)
            if not played and not env.board.game_over:
                env.board.handle_elimination()
            last_ai_move_time = current_time

        if env.board.game_over:
            winner = env.board.winner
            if winner == human_color:
                display.set_status(f"You win! ({human_color})")
            elif winner == agent_color:
                display.set_status(f"Agent wins. ({agent_color})")
            else:
                display.set_status("Tie!")

        display.set_board(env.board)
        display.render()
        clock.tick(60)

    pygame.quit()
    return env.board.winner


def watch_agent_vs_random(agent, agent_color="light", step_delay_ms=400):
    if agent_color not in ("light", "dark"):
        raise ValueError("agent_color must be 'light' or 'dark'.")
    opponent_color = "dark" if agent_color == "light" else "light"

    screen, font = _init_pygame(f"Pond - {agent.name} vs Random")
    clock = pygame.time.Clock()

    env = PondEnv()
    display = GameDisplay(screen, env.board, font, COLORS)
    display.set_status(f"{agent.name} ({agent_color}) vs Random ({opponent_color})")

    last_move_time = 0
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not env.board.game_over and current_time - last_move_time >= step_delay_ms:
            if env.board.turn == agent_color:
                played = _agent_play_turn(env, agent)
            else:
                played = _random_play_turn(env)
            if not played and not env.board.game_over:
                env.board.handle_elimination()
            last_move_time = current_time

        if env.board.game_over:
            winner = env.board.winner
            display.set_status(f"Game over - winner: {winner}")

        display.set_board(env.board)
        display.render()
        clock.tick(60)

    pygame.quit()
    return env.board.winner


def watch_random_vs_random(step_delay_ms=200):
    screen, font = _init_pygame("Pond - Random vs Random")
    clock = pygame.time.Clock()

    env = PondEnv()
    display = GameDisplay(screen, env.board, font, COLORS)
    display.set_status("Random vs Random")

    last_move_time = 0
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not env.board.game_over and current_time - last_move_time >= step_delay_ms:
            played = _random_play_turn(env)
            if not played and not env.board.game_over:
                env.board.handle_elimination()
            last_move_time = current_time

        if env.board.game_over:
            display.set_status(f"Game over - winner: {env.board.winner}")

        display.set_board(env.board)
        display.render()
        clock.tick(60)

    pygame.quit()
    return env.board.winner
