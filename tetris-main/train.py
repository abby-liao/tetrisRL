import torch
import torch.nn as nn
import numpy as np
import random
import logging
import os
import cv2
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from src.model import DQN
from src.tetris_env import TetrisEnv
from src.visualizer import TetrisVideoRecorder

LOG_NAME = "hybrid_heuristic_m1"
LOG_DIR = f"logs/{LOG_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

TOTAL_EPISODES = 12001 
BATCH_SIZE = 128
LR = 0.0003
GAMMA = 0.99
MEMORY = deque(maxlen=50000)
TARGET_UPDATE_ITER = 1000

EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_STOP_EPISODE = 8000
epsilon_step = (EPSILON_START - EPSILON_MIN) / EPSILON_STOP_EPISODE

PIECE_COLORS = {
    1: (255, 200, 0), 2: (50, 50, 255), 3: (0, 150, 255), 
    4: (0, 215, 255), 5: (50, 220, 50), 6: (200, 50, 200), 7: (255, 50, 50)
}

def render_frame(board, episode, lines, game_score, level, pieces):
    canvas = np.full((480, 420, 3), 255, dtype=np.uint8)
    for r in range(20):
        for c in range(10):
            val = int(board[r, c])
            if val > 0:
                color = PIECE_COLORS.get(val, (200, 200, 200))
                cv2.rectangle(canvas, (c*20, r*20), ((c+1)*20, (r+1)*20), color, -1)
            cv2.rectangle(canvas, (c*20, r*20), ((c+1)*20, (r+1)*20), (220, 220, 220), 1)

    font = cv2.FONT_HERSHEY_DUPLEX
    text_color = (0, 0, 0)
    cv2.putText(canvas, f"EP: {episode}", (220, 50), font, 0.7, text_color, 2)
    cv2.line(canvas, (215, 70), (400, 70), (180, 180, 180), 1)
    cv2.putText(canvas, f"LEVEL", (220, 105), font, 0.5, (100, 100, 100), 1)
    cv2.putText(canvas, f"{int(level)}", (220, 130), font, 0.7, text_color, 2)
    cv2.putText(canvas, f"LINES", (220, 175), font, 0.5, (100, 100, 100), 1)
    cv2.putText(canvas, f"{int(lines)}", (220, 200), font, 0.7, text_color, 2)
    cv2.putText(canvas, f"SCORE", (220, 245), font, 0.5, (100, 100, 100), 1)
    cv2.putText(canvas, f"{int(game_score)}", (220, 270), font, 0.7, text_color, 2)
    cv2.putText(canvas, f"PIECES", (220, 315), font, 0.5, (100, 100, 100), 1)
    cv2.putText(canvas, f"{int(pieces)}", (220, 340), font, 0.7, text_color, 2)
    cv2.rectangle(canvas, (0, 0), (200, 400), (0, 0, 0), 2)
    return canvas

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(f"{LOG_DIR}/training.log"), logging.StreamHandler()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TetrisEnv(use_render=False)
model = DQN().to(device) 
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
writer = SummaryWriter(LOG_DIR)
recorder = TetrisVideoRecorder(tb_log_dir=LOG_DIR, video_trigger=lambda ep: ep % 1000 == 0, fps=15)

epsilon = EPSILON_START
for episode in range(TOTAL_EPISODES):
    piece = env.reset()
    done, total_reward, total_lines, game_score = False, 0, 0, 0
    piece_count = 0

    while not done:
        next_steps = env.get_next_states(piece)
        if not next_steps: break
        
        actions = list(next_steps.keys())
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            model.eval()
            with torch.no_grad():
                feats_t = torch.tensor(np.array([next_steps[a][0] for a in actions]), dtype=torch.float32).to(device)
                action = actions[torch.argmax(model(feats_t)).item()]
            model.train()
        
        features, next_board = next_steps[action]
        h, l, holes, b, rt, ct = features 
        piece_count += 1

        current_level = total_lines // 10
        n_plus_1 = current_level + 1
        
        if l == 4:
            reward = 1000.0 * n_plus_1
        elif l > 0:
            reward = (l ** 2) * 15.0 * n_plus_1
        else:
            reward = 1.0 
            
        reward -= (holes * 12.0)    
        reward -= (b * 2.5)        
        reward -= (rt * 1.0)        
        reward -= (ct * 1.0)        
        reward -= (h * 0.5)         

        if np.any(next_board[0, :]):
            done = True
            reward -= 200.0

        line_scores = {1: 40, 2: 100, 3: 300, 4: 1200}
        game_score += line_scores.get(l, 0) * n_plus_1

        next_sample = [v[0] for v in env.get_next_states(random.choice(env.shapes)).values()]
        MEMORY.append((features, reward, done, next_sample))
        
        env.board = next_board
        total_reward += reward
        total_lines += l
        
        if episode % 1000 == 0:
            frame = render_frame(env.board, episode, total_lines, game_score, current_level, piece_count)
            recorder.record_frame(frame)
            
        if len(MEMORY) >= BATCH_SIZE:
            batch = random.sample(MEMORY, BATCH_SIZE)
            s_b, r_b, d_b, n_b_list = zip(*batch)
            st_t = torch.tensor(np.array(s_b), dtype=torch.float32).to(device)
            rw_t = torch.tensor(np.array(r_b), dtype=torch.float32).to(device)
            
            curr_q = model(st_t).squeeze()
            target_q = torch.zeros(BATCH_SIZE).to(device)
            
            with torch.no_grad():
                for i in range(BATCH_SIZE):
                    if d_b[i] or not n_b_list[i]:
                        target_q[i] = rw_t[i]
                    else:
                        nxt_t = torch.tensor(np.array(n_b_list[i]), dtype=torch.float32).to(device)
                        target_q[i] = rw_t[i] + GAMMA * torch.max(target_model(nxt_t))
            
            loss = criterion(curr_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        piece = random.choice(env.shapes)

    if episode % 1000 == 0:
        recorder.finalize_video(tag="Replay/Hybrid_Style", step=episode)

    if epsilon > EPSILON_MIN:
        epsilon -= epsilon_step

    if episode % TARGET_UPDATE_ITER == 0:
        target_model.load_state_dict(model.state_dict())

    if episode % 100 == 0:
        writer.add_scalar("Train/Score", game_score, episode) 
        writer.add_scalar("Train/Reward", total_reward, episode) 
        writer.add_scalar("Train/Lines", total_lines, episode)
        logging.info(f"Ep: {episode} | Lines: {total_lines} | Score: {game_score} | Eps: {epsilon:.2f}")

    if episode % 1000 == 0:
        torch.save(model.state_dict(), f"models/{LOG_NAME}_ep{episode}.pth")

writer.close()
