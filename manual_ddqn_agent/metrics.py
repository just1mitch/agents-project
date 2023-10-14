import numpy as np
import time, datetime
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

class MetricLogger():
    def __init__(self, save_dir, resume=False):
        self.save_log = save_dir / "log"

        if not resume:
            with open(self.save_log, "w") as f:
                f.write(
                    f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'Reward':>10}{'Length':>10}{'Loss':>10}{'QValue':>10}"
                    f"{'MeanReward':>15}{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}{'MaxX':>10}"
                    f"{'TimeDelta':>15}{'Time':>20}\n"
                )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        self.ep_max_xs_plot = save_dir / "max_x_plot.jpg"

        # History metrics
        self.init_data()

        # For efficient mean calculation
        self.last_100_rewards = deque(maxlen=100)
        self.last_100_lengths = deque(maxlen=100)
        self.last_100_losses = deque(maxlen=100)
        self.last_100_qs = deque(maxlen=100)

        # Timing
        
        self.record_time = time.time()
        if resume:
            self.load_previous_metrics()

    def init_data(self):
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_max_xs = []  # New field for maximum x reached
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.curr_ep_max_x = 0  # New field for maximum x reached in the current episode

    def log_step(self, reward, loss, q, max_x):  # Added max_x parameter
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1
        self.curr_ep_max_x = max(self.curr_ep_max_x, max_x)  # Update max x reached

    def log_episode(self):
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)

        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.ep_max_xs.append(self.curr_ep_max_x)

        # Efficiently update last 100 stats
        self.last_100_rewards.append(self.curr_ep_reward)
        self.last_100_lengths.append(self.curr_ep_length)
        self.last_100_losses.append(ep_avg_loss)
        self.last_100_qs.append(ep_avg_q)

        self.init_episode()

    def record(self, episode, epsilon, step):
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        # Efficient mean calculations
        mean_values = {
            "ep_rewards": np.mean(self.last_100_rewards),
            "ep_lengths": np.mean(self.last_100_lengths),
            "ep_avg_losses": np.mean(self.last_100_losses),
            "ep_avg_qs": np.mean(self.last_100_qs)
        }

        # Write to file less frequently e.g.
        if episode % 50 == 0:
            with open(self.save_log, "a") as f:
                f.write(
                    f"{episode:8d}{step:8d}{epsilon:10.3f}"
                    f"{self.ep_rewards[-1]:10.3f}{self.ep_lengths[-1]:10d}{self.ep_avg_losses[-1]:10.3f}{self.ep_avg_qs[-1]:10.3f}"
                    f"{mean_values['ep_rewards']:15.3f}{mean_values['ep_lengths']:15.3f}{mean_values['ep_avg_losses']:15.3f}{mean_values['ep_avg_qs']:15.3f}"
                    f"{self.ep_max_xs[-1]:10d}"  # Log max x for the last episode
                    f"{time_since_last_record:15.3f}"
                    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
                )

            # Plot metrics less frequently e.g., every 100 episodes
            self.plot_metrics()

    def plot_metrics(self):
        metrics = ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_max_xs"]  # Added ep_max_xs
        titles = ["Episode Rewards", "Episode Lengths", "Episode Avg Losses", "Episode Avg Qs", "Episode Max X Reached"]

        for metric, title in zip(metrics, titles):
            plt.figure(figsize=(10, 5))
            plt.plot(getattr(self, metric))
            plt.title(title)
            plt.grid(True)
            plt.ylabel(title)
            plt.xlabel("Episode")
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.close()
    def load_previous_metrics(self):
        with open(self.save_log, "r") as f:
            lines = f.readlines()[1:]  # skip the header
            for line in lines:
                data = line.strip().split()
                self.ep_rewards.append(float(data[3]))
                self.ep_lengths.append(int(data[4]))
                self.ep_avg_losses.append(float(data[5]))
                self.ep_avg_qs.append(float(data[6]))
                self.ep_max_xs.append(int(float(data[10])))
                self.last_100_rewards.append(float(data[3]))
                self.last_100_lengths.append(int(data[4]))
                self.last_100_losses.append(float(data[5]))
                self.last_100_qs.append(float(data[6]))