import torch
import cv2
import numpy as np
import random
import os
import glob
from src.model import DQN
from src.tetris_env import TetrisEnv

# Config
MODEL_DIR = "models/"
COLORS = {1:(255,255,0), 2:(255,0,0), 3:(0,165,255), 4:(0,255,255), 5:(0,255,0), 6:(128,0,128), 7:(0,0,255)}

def save_evolution_video():
    env = TetrisEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)

    # 1. Search for all model files and sort them by Episode number
    model_files = glob.glob(os.path.join(MODEL_DIR, "tetris_ep*.pth"))
    # Extracting digits to sort numerically
    model_files.sort(key=lambda x: int(x.split('ep')[-1].split('.pth')[0]))

    if not model_files:
        print("Model files not found! Please ensure 'models/' folder contains tetris_epXXXX.pth files.")
        return

    # Video Output Settings (increased height for HUD)
    video_name = 'ai_evolution.mp4'
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (350, 450))

    for model_path in model_files:
        ep_num = model_path.split('ep')[-1].split('.pth')[0]
        print(f"Recording performance for Episode: {ep_num}...")
        
        # Loading the specific checkpoint
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        piece = env.reset()
        done, score = False, 0
        step_count = 0

        # Record one game per model, capped at 500 steps to prevent infinite loops on strong AI
        while not done and step_count < 500:
            next_states = env.get_next_states(piece)
            if not next_states: break
            
            actions = list(next_states.keys())
            with torch.no_grad():
                feats = torch.stack([torch.tensor(next_states[a][0]) for a in actions]).to(device)
                action = actions[torch.argmax(model(feats)).item()]
            
            feat, env.board = next_states[action]
            score += feat[1]

            # Draw Canvas
            canvas = np.zeros((450, 350, 3), dtype=np.uint8)
            for r in range(20):
                for c in range(10):
                    if env.board[r,c] > 0:
                        cv2.rectangle(canvas, (c*20, r*20), ((c+1)*20, (r+1)*20), COLORS.get(env.board[r,c]), -1)
            
            # Display HUD Info (Heads-Up Display)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, f"TRAINING PROGRESS", (210, 30), font, 0.4, (255,255,255), 1)
            cv2.putText(canvas, f"Episode: {ep_num}", (210, 70), font, 0.6, (0, 255, 255), 2)
            cv2.putText(canvas, f"Lines: {int(score)}", (210, 110), font, 0.5, (255, 255, 255), 1)
            cv2.line(canvas, (200, 0), (200, 450), (100, 100, 100), 1)
            
            out.write(canvas)
            
            piece = random.choice(env.shapes)
            if np.any(env.board[0,:]): done = True
            step_count += 1

    out.release()
    print(f"Evolution video saved as: {video_name}")

if __name__ == "__main__":
    save_evolution_video()
