# Tetris RL Project

A reinforcement learning based Tetris agent that evaluates possible placements using a neural network and handcrafted board features.

---

##  Project Structure
(tree)


---

## Model (model.py)

This file defines a multi-layer fully connected neural network (DQN) that takes four input features and processes them through several linear layers with ReLU activations to produce a single scalar value representing the score or value of a state, allowing the system to compare different decisions. The current hidden-layer configuration (64-64-32) is just one common architectural choice, and we can also experiment with other layer sizes and combinations to evaluate their impact on training stability and performance.

---

## Environment (tetris_env.py)

This file implements a simplified Tetris environment that:

- Maintains the game board  
- Handles piece rotation, collision detection, placement, and line clearing  
- Enumerates all possible placements (x position + rotation) for a given piece  
- Returns the resulting next boards and their 4-feature state representations  
These features are used by the agent to evaluate and select the best placement.

---
## Training Pipeline (train.py)

This script implements the main training loop for the Tetris agent.

During training, the environment enumerates all possible placements (rotation and horizontal position) for the current piece. The agent evaluates each candidate placement using the neural network and selects an action using an epsilon-greedy strategy (exploration vs. exploitation).

After selecting a placement, the resulting board features and a handcrafted reward are stored in replay memory. Mini-batches are sampled from this memory to train the network via regression, allowing the model to learn a value estimate for board states.

Key responsibilities:
- Initialize the environment, model, optimizer, and replay memory  
- Enumerate candidate next states for each piece  
- Select actions using epsilon-greedy policy  
- Compute reward based on board features  
- Train the neural network using mini-batch updates  
- Log training metrics (reward, lines cleared, epsilon) to TensorBoard  
- Record gameplay GIFs periodically for visualization  
- Save model checkpoints during training

---
## record (visualizer.py)

This module provides a utility for recording gameplay frames during training and logging them to TensorBoard as animated GIFs.

The recorder collects rendered frames step-by-step throughout an episode, encodes them into a GIF using FFmpeg, and writes the result to TensorBoard’s **Images** tab. This allows visual inspection of the agent’s behavior and makes it easier to monitor learning progress, debug issues, and compare performance across episodes.

Key capabilities:
- Record frames during training  
- Encode frames into GIF format  
- Log animated gameplay to TensorBoard  
- Configure recording frequency and playback speed (FPS)

---
## record (record_video.py)
This script loads a trained Tetris model and generates a gameplay video showing how the agent plays using the learned policy.

The script initializes the Tetris environment, loads a saved model checkpoint, and repeatedly selects the best placement by evaluating all possible next states with the neural network. For each step, it renders the board into a frame, overlays the current score, and writes the frame to a video file.

The resulting output is saved as **ai_result.mp4**, providing a standalone visualization of the trained agent’s behavior without requiring TensorBoard.

---






