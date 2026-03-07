import pygame
import sys
import random
import numpy as np
from src.tetris_env import TetrisEnv

COLORS = {
    1: (255, 255, 0), 2: (255, 0, 0), 3: (0, 165, 255),
    4: (0, 255, 255), 5: (0, 255, 0), 6: (128, 0, 128), 7: (0, 0, 255)
}

class TetrisGame:
    def __init__(self):
        pygame.init()
        self.env = TetrisEnv(use_render=False)
        self.cell = 30
        self.screen = pygame.display.set_mode((self.env.width * self.cell + 200, self.env.height * self.cell))
        pygame.display.set_caption("Tetris Manual Play - Hard Drop Enabled")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.reset()

    def reset(self):
        self.env.reset()
        self.current_piece_idx = random.randint(1, 7)
        self.current_piece = self.env.shapes[self.current_piece_idx - 1]
        self.curr_pos = [0, self.env.width // 2 - len(self.current_piece[0]) // 2]
        self.total_lines = 0
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
        
        mask = np.all(self.env.board != 0, axis=1)
        num_cleared = np.sum(mask)
        if num_cleared > 0:
            new_board = self.env.board[~mask]
            self.env.board = np.vstack([np.zeros((num_cleared, self.env.width)), new_board])
            self.total_lines += num_cleared

        self.current_piece_idx = random.randint(1, 7)
        self.current_piece = self.env.shapes[self.current_piece_idx - 1]
        self.curr_pos = [0, self.env.width // 2 - len(self.current_piece[0]) // 2]
        
        if self.check_collision(self.current_piece, self.curr_pos):
            self.done = True

    def run(self):
        drop_count = 0
        while not self.done:
            self.screen.fill((20, 20, 20))
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

        print(f"Game Over! Lines: {self.total_lines}")
        pygame.quit()

    def draw(self):
        for r in range(self.env.height):
            for c in range(env_width := self.env.width):
                val = int(self.env.board[r, c])
                rect = (c * self.cell, r * self.cell, self.cell - 1, self.cell - 1)
                if val > 0:
                    pygame.draw.rect(self.screen, COLORS.get(val, (150, 150, 150)), rect)
                else:
                    pygame.draw.rect(self.screen, (40, 40, 40), rect, 1)

        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    rect = ((self.curr_pos[1] + c) * self.cell, (self.curr_pos[0] + r) * self.cell, self.cell - 1, self.cell - 1)
                    pygame.draw.rect(self.screen, COLORS.get(self.current_piece_idx), rect)

        h, _, holes, b = self.env.get_state_properties(self.env.board)
        info = [f"Lines: {int(self.total_lines)}", f"Height: {int(h)}", f"Holes: {int(holes)}", f"Bumpiness: {int(b)}"]
        for i, text in enumerate(info):
            self.screen.blit(self.font.render(text, True, (255, 255, 255)), (self.env.width * self.cell + 20, 20 + i * 30))

if __name__ == "__main__":
    TetrisGame().run()
