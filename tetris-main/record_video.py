import torch
import cv2
import numpy as np
import random
import os
import glob
import re
from src.model import DQN
from src.tetris_env import TetrisEnv

MODEL_DIR = "models/"
COLORS = {
    1: (255, 200, 0),   
    2: (50, 50, 255),   
    3: (0, 150, 255),   
    4: (0, 215, 255),   
    5: (50, 220, 50),   
    6: (200, 50, 200),  
    7: (255, 50, 50)    
}

def save_evolution_video():
    env = TetrisEnv(use_render=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)

    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    
    def get_ep_num(path):
        filename = os.path.basename(path)
        nums = re.findall(r'\d+', filename)
        return int(nums[-1]) if nums else 0

    model_files.sort(key=get_ep_num)

    if not model_files:
        print(f"No model files found in {MODEL_DIR}")
        return

    video_name = 'ai_evolution.mp4'
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (440, 450))

    for model_path in model_files:
        ep_num = get_ep_num(model_path)
        print(f"Recording Video for Episode: {ep_num}...")
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        piece = env.reset()
        done, total_lines, total_score, total_reward = False, 0, 0, 0.0
        step_count = 0

        while not done and step_count < 300:
            next_states = env.get_next_states(piece)
            if not next_states: break
            
            actions = list(next_states.keys())
            with torch.no_grad():
                feats_np = np.array([next_states[a][0] for a in actions])
                feats = torch.tensor(feats_np, dtype=torch.float32).to(device)
                action = actions[torch.argmax(model(feats)).item()]
            
            feat, env.board = next_states[action]
            h, l, holes, b = feat
            
            current_level = total_lines // 12
            n_plus_1 = current_level + 1
            
            line_scores = {1: 40, 2: 100, 3: 300, 4: 1200}
            total_score += line_scores.get(l, 0) * n_plus_1

            step_reward = 2.0
            if l == 1: step_reward += 40 * n_plus_1
            elif l == 2: step_reward += 100 * n_plus_1
            elif l == 3: step_reward += 300 * n_plus_1
            elif l == 4: step_reward += 1200 * n_plus_1
            step_reward -= (holes * 4.0)
            step_reward -= (h * 0.2)
            step_reward -= (b * 0.1)
            
            total_reward += step_reward
            total_lines += l

            canvas = np.full((450, 440, 3), 255, dtype=np.uint8)
            for r in range(20):
                for c in range(10):
                    cv2.rectangle(canvas, (c*20, r*20), ((c+1)*20, (r+1)*20), (230, 230, 230), 1)
                    val = int(env.board[r,c])
                    if val > 0:
                        cv2.rectangle(canvas, (c*20, r*20), ((c+1)*20, (r+1)*20), COLORS.get(val, (200,200,200)), -1)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            text_color = (0, 0, 0)
            
            cv2.putText(canvas, f"EP: {ep_num}", (220, 50), font, 0.7, text_color, 2)
            cv2.line(canvas, (215, 70), (420, 70), (180, 180, 180), 1)
            
            cv2.putText(canvas, f"LEVEL", (220, 110), font, 0.5, (100, 100, 100), 1)
            cv2.putText(canvas, f"{int(current_level)}", (220, 140), font, 0.8, text_color, 2)
            
            cv2.putText(canvas, f"LINES", (220, 190), font, 0.5, (100, 100, 100), 1)
            cv2.putText(canvas, f"{int(total_lines)}", (220, 220), font, 0.8, text_color, 2)
            
            cv2.putText(canvas, f"SCORE", (220, 270), font, 0.5, (100, 100, 100), 1)
            cv2.putText(canvas, f"{int(total_score)}", (220, 300), font, 0.8, text_color, 2)

            cv2.putText(canvas, f"REWARD", (220, 350), font, 0.5, (0, 120, 0), 1)
            cv2.putText(canvas, f"{total_reward:.1f}", (220, 380), font, 0.8, (0, 150, 0), 2)
            
            cv2.rectangle(canvas, (0, 0), (200, 400), (0, 0, 0), 2)
            
            out.write(canvas)
            
            piece = random.choice(env.shapes)
            if np.any(env.board[0,:]): 
                total_reward -= 50
                done = True
            step_count += 1

    out.release()
    print(f"Success! Evolution video saved as: {video_name}")

if __name__ == "__main__":
    save_evolution_video()
