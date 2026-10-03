# Deep Q-Learning for Lunar Landing

This project trains a Deep Q-Network (DQN) agent to solve the Gymnasium LunarLander environment. The notebook implements a reinforcement learning pipeline with experience replay, a target network, epsilon-greedy exploration, and a video rendering helper for visualizing the trained policy.

## Project overview

The agent learns to control a lunar lander by observing its state and choosing one of several discrete actions. The implementation follows a classic DQN setup:

- Input: LunarLander observation vector
- Network: fully connected neural network
- Output: Q-values for each action
- Training: Bellman target optimization with replay memory
- Exploration: epsilon-greedy strategy
- Stabilization: target network with soft updates

## Files in this workspace

- `Deep_Q_Learning_for_Lunar_Landing_Complete_Code.ipynb` — full implementation notebook
- `Deep Q-Learning for Lunar Landing - Complete Code.ipynb` — alternate notebook name/variant

## Environment and dependencies

The notebook installs and uses:

- Python 3
- PyTorch
- NumPy
- Gymnasium
- Box2D support for Lunar Lander
- imageio for video generation

Install the required packages:

```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
apt-get install -y swig
pip install gymnasium[box2d]
```

## Model architecture

The neural network is defined as a 3-layer MLP:

```python
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
```

This agent estimates Q-values for each possible action and selects the action with the highest value during evaluation.

## Training setup

The notebook configures the environment and key hyperparameters as follows:

```python
import gymnasium as gym
env = gym.make('LunarLander-v3')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
```

```python
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.50
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3
```

### Replay memory

The project stores transitions in a replay buffer:

- state
- action
- reward
- next state
- done flag

This allows the model to train from a diverse set of past experiences and reduces correlations between consecutive samples.

## Agent logic

The agent implements:

- epsilon-greedy action selection
- experience replay sampling
- Q-learning update
- target network soft update

Core update rule:

```python
next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
q_expected = self.local_qnetwork(states).gather(1, actions)
loss = F.mse_loss(q_expected, q_targets)
```

The network is updated with Adam, and the target network is gradually blended in using the interpolation parameter.

## Training loop

The training loop uses:

- `number_episodes = 2000`
- `maximum_number_timesteps_per_episode = 1000`
- `epsilon_starting_value = 1.0`
- `epsilon_ending_value = 0.01`
- `epsilon_decay_value = 0.995`

At each episode:

1. Reset the environment
2. Choose an action using epsilon-greedy policy
3. Step the environment
4. Store the transition in replay memory
5. Learn from a batch every few steps
6. Decay epsilon over time
7. Print average reward statistics every 100 episodes

The training stops when the rolling 100-episode average score reaches 200 or more:

```python
if np.mean(scores_on_100_episodes) >= 200.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(...))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break
```

## Result visualization

The notebook includes a helper to render the trained model as a video:

```python
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)
```

This creates a saved MP4 and displays it in the notebook.

## How to run the project

1. Open the notebook in Jupyter or VS Code.
2. Run the installation cells.
3. Run the imports and network definition cells.
4. Initialize the environment and agent.
5. Execute the training loop.
6. Optionally render the trained agent with the video helper.

## Summary

This project demonstrates a practical implementation of DQN for a classic reinforcement learning challenge. It shows the full workflow from environment setup and model construction to experience replay training and policy evaluation in the LunarLander task.

## License

This project is provided for educational purposes and is intended for learning and experimentation with reinforcement learning.
