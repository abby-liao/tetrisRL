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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler("training_standard.log"), logging.StreamHandler()])

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
writer = SummaryWriter("logs/tetris_standard_greedy")

recorder = TetrisVideoRecorder(
    tb_log_dir="logs/tetris_standard_greedy", 
    video_trigger=lambda ep: ep % 500 == 0,
    fps=15
)

PIECE_COLORS = {
    1: (255, 255, 0), 2: (255, 0, 0), 3: (0, 165, 255), 
    4: (0, 255, 255), 5: (0, 255, 0), 6: (128, 0, 128), 7: (0, 0, 255)      
}

BATCH_SIZE = 32
LR = 0.01
GAMMA = 0.99
MEMORY = deque(maxlen=30000)
TOTAL_EPISODES = 15001 

EPSILON = 1.0           # Initial exploration rate
EPSILON_MIN = 0.01      # Minimum exploration rate
EPSILON_DECAY = 0.999   # Decay factor per episode (Standard greedy logic)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TetrisEnv(use_render=False)
model = DQN().to(device)
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
    cv2.putText(canvas, "STANDARD GREEDY", (210, 30), font, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, f"Ep: {episode}", (210, 70), font, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, f"Lines: {int(score)}", (210, 110), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"Reward: {int(reward)}", (210, 150), font, 0.5, (100, 100, 255), 1)
    cv2.putText(canvas, f"Eps: {epsilon:.3f}", (210, 190), font, 0.5, (100, 255, 100), 1)
    cv2.line(canvas, (200, 0), (200, 450), (255, 255, 255), 1)
    return canvas

def train_step():
    if len(MEMORY) < BATCH_SIZE: return 0
    batch = random.sample(MEMORY, BATCH_SIZE)
    states, rewards, done_flags = zip(*batch)
    state_t = torch.tensor(np.array(states)).to(device)
    reward_t = torch.tensor(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(device)
    
    q_values = model(state_t)
    loss = criterion(q_values, reward_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

epsilon = EPSILON

for episode in range(TOTAL_EPISODES):
    piece = env.reset()
    done = False
    total_lines = 0
    total_reward = 0

    while not done:
        next_steps = env.get_next_states(piece)
        if not next_steps: break
        
        actions = list(next_steps.keys())
        
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            model.eval()
            with torch.no_grad():
                feats = torch.stack([torch.tensor(next_steps[a][0]) for a in actions]).to(device)
                preds = model(feats)
                action = actions[torch.argmax(preds).item()]
            model.train()
        
        best_feat, best_board = next_steps[action]
        h, l, holes, b = best_feat 
        
        reward = (l**2 * 12.0) - (h * 0.25) - (holes * 10.0) - (b * 0.2) + 2.0
        
        if np.any(best_board[0, :]):
            reward -= 50 
            done = True

        MEMORY.append((best_feat, reward, done))
        env.board = best_board
        total_reward += reward
        total_lines += l
        
        if episode % 500 == 0:
            frame = render_frame(env.board, episode, total_lines, total_reward, epsilon)
            recorder.record_frame(frame)

        train_step()
        piece = random.choice(env.shapes)
    
    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY
        epsilon = max(epsilon, EPSILON_MIN)

    if episode % 50 == 0: 
        writer.add_scalar("Train/Reward", total_reward, episode)
        writer.add_scalar("Train/Lines_Cleared", total_lines, episode)
        writer.add_scalar("Train/Epsilon", epsilon, episode)
        logging.info(f"Episode: {episode} | Lines: {total_lines} | Reward: {total_reward:.1f} | Eps: {epsilon:.3f}")

    if episode % 500 == 0:
        recorder.finalize_video(tag="Replay/Animated_GIF", step=episode)
        torch.save(model.state_dict(), f"models/tetris_standard_ep{episode}.pth")

writer.close()
print("Training Complete. Standard Epsilon-Greedy models saved.")
