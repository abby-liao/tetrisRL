# This class implements a simplified Tetris environment used for training an agent.
#
# Main responsibilities:
# - Maintain the game board state (10x20 grid)
# - Provide the set of tetromino shapes
# - Handle piece rotation
# - Detect collisions with the board or boundaries
# - Place pieces onto the board
# - Clear completed lines
#
# Core functionality:
# Given a current piece, the environment enumerates all possible placements
# (every rotation and valid x position), simulates dropping the piece, and
# returns the resulting next board states along with their corresponding
# 4-feature representations.
#
# These features allow the agent (neural network) to evaluate each candidate
# placement and choose the best action.

import pygame
import numpy as np
import random
import os

class TetrisEnv:
    def __init__(self, use_render=False):
        if not use_render:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        pygame.init()
        self.width, self.height = 10, 20
        self.block_size = 30
        self.board = np.zeros((self.height, self.width), dtype=int)
        
        self.shapes = [
            [[1, 1, 1, 1]],         # I (1)
            [[2, 2], [2, 2]],       # O (2)
            [[0, 3, 0], [3, 3, 3]], # T (3)
            [[4, 4, 0], [0, 4, 4]], # Z (4)
            [[0, 5, 5], [5, 5, 0]], # S (5)
            [[6, 0, 0], [6, 6, 6]], # J (6)
            [[0, 0, 7], [7, 7, 7]]  # L (7)
        ]
        
        if use_render:
            self.screen = pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        return random.choice(self.shapes)

    def rotate(self, shape):
        return [list(row) for row in zip(*shape[::-1])]

    def get_state_properties(self, board):
        heights = np.zeros(self.width)
        for c in range(self.width):
            col = board[:, c]
            if np.any(col > 0):
                heights[c] = self.height - np.argmax(col > 0)
        
        agg_height = np.sum(heights)
        bumpiness = np.sum(np.abs(np.diff(heights)))
        holes = 0
        for c in range(self.width):
            col = board[:, c]
            if np.any(col > 0):
                first_block = np.argmax(col > 0)
                holes += np.sum(col[first_block:] == 0)
        
        return np.array([agg_height, 0, holes, bumpiness], dtype=np.float32)

    def get_next_states(self, shape):
        states = {}
        curr_shape = shape
        for r in range(4):
            for x in range(self.width - len(curr_shape[0]) + 1):
                temp_board = self.board.copy()
                y = 0
                while y + len(curr_shape) <= self.height:
                    if self.check_collision(temp_board, curr_shape, x, y):
                        break
                    y += 1
                y -= 1
                
                if y < 0: continue
                
                self.add_to_board(temp_board, curr_shape, x, y)
                lines = self.clear_lines(temp_board)
                features = self.get_state_properties(temp_board)
                features[1] = lines 
                states[(x, r)] = (features, temp_board)
            
            curr_shape = self.rotate(curr_shape)
        return states

    def check_collision(self, board, shape, x, y):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell > 0: 
                    if (y + r >= self.height or 
                        x + c >= self.width or 
                        x + c < 0 or 
                        board[y + r, x + c] > 0):
                        return True
        return False

    def add_to_board(self, board, shape, x, y):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell > 0:
                    board[y + r, x + c] = cell

    def clear_lines(self, board):
        full_rows = np.all(board > 0, axis=1)
        num_full = np.sum(full_rows)
        if num_full > 0:
            non_full_rows = board[~full_rows]
            new_board = np.zeros_like(board)
            new_board[self.height - len(non_full_rows):] = non_full_rows
            board[:] = new_board
        return num_full
