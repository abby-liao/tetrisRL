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
### Resources
-Reward Based Epsilon Decay
-https://aakash94.github.io/Reward-Based-Epsilon-Decay/
Reward-Based Epsilon Decay (RBED) is an exploration strategy in reinforcement learning that adjusts the ε value in ε-greedy policies based on the agent’s performance rather than time or episode count. Instead of gradually decreasing ε on a fixed schedule, RBED lowers ε only when the agent reaches a predefined reward threshold, then raises the threshold for the next stage. This creates a performance-driven transition from exploration to exploitation, ensuring that the agent reduces exploration only after demonstrating learning progress. As a result, the approach can produce more stable training, better reproducibility, and more intuitive hyperparameter tuning, although its effectiveness depends on the quality and consistency of reward signals in a given environment.

-Introducing Q Learning
-https://huggingface.co/learn/deep-rl-course/en/unit2/q-learning

-Batch Size
-https://ai.stackexchange.com/questions/23254/is-there-a-logical-method-of-deducing-an-optimal-batch-size-when-training-a-deep
There is no universal method to derive the optimal batch size for DQN. In practice, 32 or 64 are commonly used as default values, while larger batch sizes can be explored when aiming for the best performance. Ultimately, the optimal batch size is task-dependent and must be determined through experimentation.

-Hidden layer
-https://www.heatonresearch.com/2017/06/01/hidden-layers.html


-Gamma/ LR parameter settings reference
-https://codesignal.com/learn/courses/q-learning-unleashed-building-intelligent-agents/lessons/introduction-to-q-learning-building-intelligent-agents

---






