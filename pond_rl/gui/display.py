import pygame


COLORS = {
    "WHITE": (245, 245, 250),
    "BLACK": (20, 20, 30),
    "GRID": (40, 40, 60),
    "LIGHT_COLOR": (200, 200, 255),
    "DARK_COLOR": (80, 80, 180),
    "HIGHLIGHT_COLOR": (255, 220, 70),
    "TEXT": (30, 30, 40),
    "PANEL": (220, 220, 235),
}


class GameDisplay:
    CELL_SIZE = 100
    GRID_TOP = 130
    WINDOW_WIDTH = 400
    WINDOW_HEIGHT = 600

    def __init__(self, screen, board, font, colors=None):
        self.screen = screen
        self.board = board
        self.font = font
        self.colors = colors or COLORS
        self.selected_piece = None
        self.status_message = ""

    def set_board(self, board):
        self.board = board
        self.selected_piece = None

    def set_status(self, message):
        self.status_message = message

    def draw_grid(self):
        for col in range(self.board.grid_size):
            for row in range(self.board.grid_size):
                x = col * self.CELL_SIZE
                y = self.GRID_TOP + row * self.CELL_SIZE
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.colors["PANEL"], rect)
                pygame.draw.rect(self.screen, self.colors["GRID"], rect, 2)

    def draw_pieces(self):
        for row in range(self.board.grid_size):
            for col in range(self.board.grid_size):
                piece = self.board.grid[row][col]
                if piece is None:
                    continue
                color = self.colors["LIGHT_COLOR"] if piece.color == "light" else self.colors["DARK_COLOR"]
                center_x = col * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_TOP + row * self.CELL_SIZE + self.CELL_SIZE // 2
                if self.selected_piece == (row, col):
                    pygame.draw.circle(self.screen, self.colors["HIGHLIGHT_COLOR"], (center_x, center_y), 40)
                pygame.draw.circle(self.screen, color, (center_x, center_y), 30)
                symbol = piece.get_piece_type()[0]
                text = self.font.render(symbol, True, self.colors["BLACK"])
                text_rect = text.get_rect(center=(center_x, center_y))
                self.screen.blit(text, text_rect)

    def draw_score(self):
        light_score_text = self.font.render(
            f"Light: {self.board.light_player.score}", True, self.colors["DARK_COLOR"]
        )
        dark_score_text = self.font.render(
            f"Dark: {self.board.dark_player.score}", True, self.colors["DARK_COLOR"]
        )
        light_tokens_text = self.font.render(
            f"Tokens: {self.board.light_player.remaining_tokens()}", True, self.colors["TEXT"]
        )
        dark_tokens_text = self.font.render(
            f"Tokens: {self.board.dark_player.remaining_tokens()}", True, self.colors["TEXT"]
        )
        turn_text = self.font.render(f"Turn: {self.board.turn.capitalize()}", True, self.colors["TEXT"])
        self.screen.blit(light_score_text, (10, 10))
        self.screen.blit(dark_score_text, (220, 10))
        self.screen.blit(light_tokens_text, (10, 50))
        self.screen.blit(dark_tokens_text, (220, 50))
        self.screen.blit(turn_text, (140, 90))

    def draw_status(self):
        if not self.status_message:
            return
        text = self.font.render(self.status_message, True, self.colors["BLACK"])
        text_rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, self.GRID_TOP + 4 * self.CELL_SIZE + 30))
        self.screen.blit(text, text_rect)

    def render(self):
        self.screen.fill(self.colors["WHITE"])
        self.draw_score()
        self.draw_grid()
        self.draw_pieces()
        self.draw_status()
        pygame.display.flip()

    def cell_from_mouse(self, x, y):
        if y < self.GRID_TOP:
            return None
        row = (y - self.GRID_TOP) // self.CELL_SIZE
        col = x // self.CELL_SIZE
        if 0 <= row < self.board.grid_size and 0 <= col < self.board.grid_size:
            return int(row), int(col)
        return None

    def handle_human_click(self, env, human_color):
        if self.board.game_over or self.board.turn != human_color:
            return None
        x, y = pygame.mouse.get_pos()
        cell = self.cell_from_mouse(x, y)
        if cell is None:
            return None
        row, col = cell

        env.available_actions()
        valid_action_ids = set()
        for idx, present in enumerate(env.action_mask):
            if present:
                valid_action_ids.add(idx)

        if self.selected_piece is None:
            if self.board.grid[row][col] is None:
                placement_action = {"type": "place", "row": row, "col": col, "piece": "Egg"}
                placement_id = env.get_action_id(placement_action)
                if placement_id in valid_action_ids:
                    return placement_id
                return None
            piece = self.board.grid[row][col]
            if piece.color == human_color:
                self.selected_piece = (row, col)
            return None

        start_row, start_col = self.selected_piece
        if (row, col) == (start_row, start_col):
            self.selected_piece = None
            return None
        move_action = {
            "type": "move",
            "start_row": start_row,
            "start_col": start_col,
            "end_row": row,
            "end_col": col,
        }
        try:
            move_id = env.get_action_id(move_action)
        except ValueError:
            self.selected_piece = None
            return None
        if move_id in valid_action_ids:
            self.selected_piece = None
            return move_id
        self.selected_piece = None
        return None
