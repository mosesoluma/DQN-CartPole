import numpy as np

np.bool8 = np.bool_ 

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 2e-4    # Matches Reference Blog Text
GAMMA = 0.99            # Discount Factor
HIDDEN_SIZE = 64        # Network Size
BUFFER_SIZE = 10000     # Replay Buffer
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10      # Update target network every 10 episodes

# --- Device Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. The Q-Network ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 3 Layers (Input -> 64 -> 64 -> Output) as per assignment reference
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- 3. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)).to(device),
                torch.LongTensor(action).unsqueeze(1).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_state)).to(device),
                torch.FloatTensor(done).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

# --- 4. Training Function ---
def train_dqn():
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPSILON_START
    scores = []

    print(f"Starting training on {ENV_NAME} with Reward Shaping (-100 penalty)...")

    for episode in range(1, 1001):
        # --- ROBUST RESET (Handles old & new Gym versions) ---
        reset_return = env.reset()
        if isinstance(reset_return, tuple):
            state = reset_return[0] 
        else:
            state = reset_return    
        
        score = 0
        done = False
        
        while not done:
            # Epsilon-Greedy Action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax(dim=1).item()

            # --- ROBUST STEP (Handles old & new Gym versions) ---
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

           
            if done and score < 499:
                reward = -100
            # -------------------------------------------

            memory.push(state, action, reward, next_state, done)
            state = next_state
            score += 1

            # Optimization
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # Q(s,a)
                q_values = policy_net(states).gather(1, actions)
                
                # Target Q
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Epsilon Decay
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        scores.append(score)

        # Update Target Network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Progress Logging
        if episode % 20 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}\tScore: {score}\tAvg Score (100): {avg_score:.2f}\tEpsilon: {epsilon:.2f}")

        # Stop if solved (Average > 475)
        if np.mean(scores[-100:]) > 475:
            print(f"Solved in {episode} episodes!")
            break

    env.close()
    return scores

# --- 5. Plotting Function ---
def plot_results(scores):
    moving_avg = []
    for i in range(len(scores)):
        window = scores[max(0, i-100):i+1]
        moving_avg.append(sum(window) / len(window))

    plt.figure(figsize=(10, 5))
    plt.plot(scores, color='lightblue', alpha=0.5, label='Raw Score')
    plt.plot(moving_avg, color='blue', label='Avg Score (100 eps)')
    plt.title('DQN CartPole-v1: Reward Shaping (-100 Penalty)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_final_result.png')
    plt.show()

if __name__ == "__main__":
    scores = train_dqn()
    plot_results(scores)