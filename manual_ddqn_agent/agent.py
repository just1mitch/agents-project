import torch
import random
import numpy as np
from pathlib import Path
from collections import deque
from neural import MarioNet

# In the original code, the agent is trained for 80000 episodes, but we will only train for 10000 episodes
#https://github.com/yfeng997/MadMario/blob/master/agent.py
class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        """Initialize Mario agent."""

        # Environment and DDQN parameters - https://github.com/mrshenli/rpc-rl-experiments/blob/main/agent.py

        self.state_dim = state_dim  # dimensions of the env
        self.action_dim = action_dim  # possible actions

        self.memory = deque(maxlen=100000)  # Replay buffer
        self.batch_size = 64  # Number of experiences to sample

        self.exploration_rate = 1  # Initial exploration rate for the epsilon-greedy policy.
        self.exploration_rate_decay = 0.99999975 #0.99999975  # Decay rate for exploratrion prob - lower the epsilon as it learns to exploit rather than explore
        self.exploration_rate_min = 0.1  # Minimum exploration rate to ensure some level of exploration.

        self.gamma = 0.9  # Discount factor for future rewards in Q-learning. [Most sources have this 0.9 - 0.99]
        self.curr_step = 0  # Current step or iteration of the agent.
        self.current_episode = 0  # Current episode of the agent.
        
        self.burnin = 10000  # Number of steps before the agent starts training.


        self.learn_every = 3  # Frequency (in steps) at which the agent learns from experiences.
        self.tau = 0.005 # Soft update of target network - the lower the value, the slower the target network will be updated
        self.sync_every = 1.2e4  # Frequency (in steps) at which the target Q-network is updated with the online Q-network's weights.
        self.save_every = 50000  # Frequency (in steps) at which the agent's model is saved.
        self.save_dir = save_dir
        self.use_cuda = torch.cuda.is_available()

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        # self.net = MarioNet(self.state_dim, self.action_dim).float().cuda()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0003) #0.00025
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """Choose an epsilon-greedy action."""
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_rate:
            desired = np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).cuda().unsqueeze(0)
            action_values = self.net(state, model='online')
            desired = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return desired

    def cache(self, state, next_state, action, reward, done):
        """Store experience in memory."""
        # Save the experience to our episode buffer.
        
        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.LongTensor([action]).cuda()
        reward = torch.DoubleTensor([reward]).cuda()
        done = torch.BoolTensor([done]).cuda()
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """Retrieve a batch of experiences from memory."""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """Get TD Estimate."""
        return self.net(state, model='online')[np.arange(0, self.batch_size), action]


# https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
# Torch.no_grad() is a context manager that disables gradient calculation.
# This will reduce memory consumption for computations that would otherwise have requires_grad=True.

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """Get TD Target."""
        take_next_Q = self.net(next_state, model='online')
        best_action = torch.argmax(take_next_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """Update Q online value."""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping - used to prevent exploding gradients in very deep networks
        for param in self.net.online.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Sync Q target value."""
        # Implemented as soft update - this should speed up training
        for target_param, online_param in zip(self.net.target.parameters(), self.net.online.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self):
        """Mario learns from experience."""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save(episode=self.current_episode)
        if self.curr_step < self.burnin or self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    def save(self, episode):
        """Save MarioNet state."""
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save({
            'model': self.net.state_dict(),
            'exploration_rate': self.exploration_rate,
            'episode': episode,  # Save the episode number so we can re-train
            'curr_step': self.curr_step
        }, save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        """Load MarioNet state."""
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")
        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        self.net.load_state_dict(ckp.get('model'))
        self.exploration_rate = ckp.get('exploration_rate')
        self.curr_step = ckp.get('curr_step', 0)
        episode = ckp.get('episode', 0)  # Load the episode number, default to 0 if not present
        print(f"Loading model at {load_path} with exploration rate {self.exploration_rate}")
        return episode
