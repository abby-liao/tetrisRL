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

BATCH_SIZE = 128
LR = 0.0001 
GAMMA = 0.99
MEMORY = deque(maxlen=50000)
TARGET_UPDATE_ITER = 1000 
EPSILON_DECAY = 0.9998
EPSILON_MIN = 0.005

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler("training_dqn_4feat.log"), logging.StreamHandler()])

os.makedirs("logs/dqn_4feat", exist_ok=True)
os.makedirs("models", exist_ok=True)

writer = SummaryWriter("logs/dqn_4feat")
recorder = TetrisVideoRecorder(
    tb_log_dir="logs/dqn_4feat", 
    video_trigger=lambda ep: ep % 500 == 0, 
    fps=15
)

PIECE_COLORS = {1: (255, 255, 0), 2: (255, 0, 0), 3: (0, 165, 255), 4: (0, 255, 255), 5: (0, 255, 0), 6: (128, 0, 128), 7: (0, 0, 255)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TetrisEnv(use_render=False)

model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

def render_frame(board, episode, score, reward, epsilon):
    canvas = np.zeros((450, 350, 3), dtype=np.uint8)
    for r in range(20):
        for c in range(10):
            val = int(board[r, c])
            if val > 0:
                color = PIECE_COLORS.get(val, (255, 255, 255))
                cv2.rectangle(canvas, (c*20, r*20), ((c+1)*20, (r+1)*20), color, -1)
            cv2.rectangle(canvas, (c*20, r*20), ((c+1)*20, (r+1)*20), (40, 40, 40), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Ep: {episode}", (210, 70), font, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, f"Score: {int(score)}", (210, 110), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"Eps: {epsilon:.3f}", (210, 150), font, 0.5, (100, 255, 100), 1)
    return canvas

def train_step():
    if len(MEMORY) < BATCH_SIZE: return 0
    
    batch = random.sample(MEMORY, BATCH_SIZE)
    states, rewards, dones, next_states_list = zip(*batch)

    state_t = torch.tensor(np.array(states)).to(device)
    reward_t = torch.tensor(np.array(rewards, dtype=np.float32)).to(device)
    done_t = torch.tensor(np.array(dones, dtype=np.float32)).to(device)

    current_q = model(state_t).squeeze()

    target_q = torch.zeros(BATCH_SIZE).to(device)
    with torch.no_grad():
        for i in range(BATCH_SIZE):
            if dones[i] or not next_states_list[i]:
                target_q[i] = reward_t[i]
            else:
                next_feats_t = torch.tensor(np.array(next_states_list[i])).to(device)
                next_q_values = target_model(next_feats_t)
                target_q[i] = reward_t[i] + GAMMA * torch.max(next_q_values)

    loss = criterion(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

epsilon = 1.0

for episode in range(150000):
    piece = env.reset()
    done = False
    total_reward = 0
    total_lines = 0

    while not done:
        next_steps = env.get_next_states(piece)
        if not next_steps: break
        
        actions = list(next_steps.keys())
        
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            model.eval()
            with torch.no_grad():
                feats_t = torch.tensor(np.array([next_steps[a][0] for a in actions])).to(device)
                preds = model(feats_t)
                action = actions[torch.argmax(preds).item()]
            model.train()
        
        best_feat, best_board = next_steps[action]
        h, l, holes, b = best_feat

        reward = 1.0 
        if l > 0:
            reward += (l ** 2) * 100.0 + 50.0

        reward -= (holes * 5.0)
        reward -= (h * 0.5)
        reward -= (b * 0.2)

        if np.any(best_board[0, :]):
            done = True
            reward -= 50

        next_piece = random.choice(env.shapes)
        next_possible_steps = env.get_next_states(next_piece)
        next_state_feats = [v[0] for v in next_possible_steps.values()]

        MEMORY.append((best_feat, reward, done, next_state_feats))
        
        env.board = best_board
        total_reward += reward
        total_lines += l
        
        if episode % 500 == 0:
            frame = render_frame(env.board, episode, total_lines, total_reward, epsilon)
            recorder.record_frame(frame)

        train_step()
        piece = next_piece

    if episode % 500 == 0:
        recorder.finalize_video(tag="Replay/Animated_GIF", step=episode)

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE_ITER == 0:
        target_model.load_state_dict(model.state_dict())

    if episode % 50 == 0:
        writer.add_scalar("Train/Reward", total_reward, episode)
        writer.add_scalar("Train/Lines", total_lines, episode)
        writer.add_scalar("Train/Epsilon", epsilon, episode)
        writer.flush()
        logging.info(f"Ep: {episode} | Lines: {total_lines} | Reward: {total_reward:.1f}")

    if episode % 500 == 0:
        torch.save(model.state_dict(), f"models/tetris_4feat_ep{episode}.pth")

writer.close()
