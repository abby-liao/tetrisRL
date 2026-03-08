import pygame
import sys
import random
import numpy as np
from src.tetris_env import TetrisEnv

COLORS = {
    1: (255, 200, 0),   
    2: (50, 50, 255),   
    3: (0, 150, 255),   
    4: (0, 215, 255),   
    5: (50, 220, 50),   
    6: (200, 50, 200),  
    7: (255, 50, 50)    
}

class TetrisGame:
    def __init__(self):
        pygame.init()
        self.env = TetrisEnv(use_render=False)
        self.cell = 30
        self.screen = pygame.display.set_mode((self.env.width * self.cell + 220, self.env.height * self.cell))
        pygame.display.set_caption("Tetris Manual Play - Score & Reward")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.reset()

    def reset(self):
        self.env.reset()
        self.current_piece_idx = random.randint(1, 7)
        self.current_piece = self.env.shapes[self.current_piece_idx - 1]
        self.curr_pos = [0, self.env.width // 2 - len(self.current_piece[0]) // 2]
        self.total_lines = 0
        self.total_score = 0
        self.total_reward = 0.0
        self.done = False

    def check_collision(self, piece, pos):
        for r, row in enumerate(piece):
            for c, val in enumerate(row):
                if val:
                    new_r, new_c = pos[0] + r, pos[1] + c
                    if new_r >= self.env.height or new_c < 0 or new_c >= self.env.width or self.env.board[new_r, new_c]:
                        return True
        return False

    def lock_piece(self):
        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    self.env.board[self.curr_pos[0] + r, self.curr_pos[1] + c] = self.current_piece_idx
        
        feat, _ = self.env.get_state_properties(self.env.board)
        h, num_cleared, holes, b = feat
        
        current_level = self.total_lines // 12
        n_plus_1 = current_level + 1
        
        line_scores = {1: 40, 2: 100, 3: 300, 4: 1200}
        self.total_score += line_scores.get(num_cleared, 0) * n_plus_1

        step_reward = 2.0
        if num_cleared == 1: step_reward += 40 * n_plus_1
        elif num_cleared == 2: step_reward += 100 * n_plus_1
        elif num_cleared == 3: step_reward += 300 * n_plus_1
        elif num_cleared == 4: step_reward += 1200 * n_plus_1
        
        step_reward -= (holes * 4.0)
        step_reward -= (h * 0.2)
        step_reward -= (b * 0.1)
        
        self.total_reward += step_reward

        mask = np.all(self.env.board != 0, axis=1)
        if num_cleared > 0:
            new_board = self.env.board[~mask]
            self.env.board = np.vstack([np.zeros((num_cleared, self.env.width)), new_board])
            self.total_lines += num_cleared

        self.current_piece_idx = random.randint(1, 7)
        self.current_piece = self.env.shapes[self.current_piece_idx - 1]
        self.curr_pos = [0, self.env.width // 2 - len(self.current_piece[0]) // 2]
        
        if self.check_collision(self.current_piece, self.curr_pos):
            self.total_reward -= 50
            self.done = True

    def run(self):
        drop_count = 0
        while not self.done:
            self.screen.fill((255, 255, 255))
            dt = self.clock.tick(60)
            drop_count += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        if not self.check_collision(self.current_piece, [self.curr_pos[0], self.curr_pos[1] - 1]):
                            self.curr_pos[1] -= 1
                    if event.key == pygame.K_RIGHT:
                        if not self.check_collision(self.current_piece, [self.curr_pos[0], self.curr_pos[1] + 1]):
                            self.curr_pos[1] += 1
                    if event.key == pygame.K_DOWN:
                        if not self.check_collision(self.current_piece, [self.curr_pos[0] + 1, self.curr_pos[1]]):
                            self.curr_pos[0] += 1
                    if event.key == pygame.K_UP:
                        rotated = np.rot90(self.current_piece)
                        if not self.check_collision(rotated, self.curr_pos):
                            self.current_piece = rotated
                    if event.key == pygame.K_SPACE:
                        while not self.check_collision(self.current_piece, [self.curr_pos[0] + 1, self.curr_pos[1]]):
                            self.curr_pos[0] += 1
                        self.lock_piece()
                        drop_count = 0

            if drop_count > 500:
                if not self.check_collision(self.current_piece, [self.curr_pos[0] + 1, self.curr_pos[1]]):
                    self.curr_pos[0] += 1
                else:
                    self.lock_piece()
                drop_count = 0

            self.draw()
            pygame.display.flip()

        pygame.quit()

    def draw(self):
        for r in range(self.env.height):
            for c in range(self.env.width):
                val = int(self.env.board[r, c])
                rect = (c * self.cell, r * self.cell, self.cell - 1, self.cell - 1)
                if val > 0:
                    pygame.draw.rect(self.screen, COLORS.get(val, (200, 200, 200)), rect)
                else:
                    pygame.draw.rect(self.screen, (230, 230, 230), rect, 1)

        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    rect = ((self.curr_pos[1] + c) * self.cell, (self.curr_pos[0] + r) * self.cell, self.cell - 1, self.cell - 1)
                    pygame.draw.rect(self.screen, COLORS.get(self.current_piece_idx), rect)

        level = self.total_lines // 12
        info = [
            f"LEVEL: {int(level)}",
            f"LINES: {int(self.total_lines)}",
            f"SCORE: {int(self.total_score)}",
            f"REWARD: {self.total_reward:.1f}"
        ]
        
        for i, text in enumerate(info):
            color = (0, 0, 0) if "REWARD" not in text else (0, 150, 0)
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (self.env.width * self.cell + 20, 40 + i * 50))
        
        pygame.draw.line(self.screen, (0, 0, 0), (self.env.width * self.cell, 0), (self.env.width * self.cell, self.env.height * self.cell), 2)

if __name__ == "__main__":
    TetrisGame().run()
