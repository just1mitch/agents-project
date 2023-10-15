import pandas as pd
import matplotlib.pyplot as plt
import argparse

class LogAnalyzer:
    def __init__(self, log_file_path):
        self.df = pd.read_csv(log_file_path, delim_whitespace=True)

    def basic_statistics(self):
        """Print basic statistics."""
        print(self.df.describe())

    def plot_metric(self, metric, smoothing_window=0):
        """Plot the specified metric with optional smoothing."""
        plt.figure(figsize=(10, 5))
        if smoothing_window > 0:
            smoothed_data = self.df[metric].rolling(window=smoothing_window).mean()
            plt.plot(self.df.index, smoothed_data, label=f'Smoothed ({smoothing_window} episodes)')
        plt.plot(self.df.index, self.df[metric], alpha=0.4, label='Raw')
        plt.title(metric)
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze(self, smoothing_window=0):
        metrics_to_plot = ["Reward", "Length", "Loss", "QValue", "MeanReward", "MeanLength", "MeanLoss", "MeanQValue", "MaxX"]
        self.basic_statistics()
        for metric in metrics_to_plot:
            self.plot_metric(metric, smoothing_window=smoothing_window)

def main():
    parser = argparse.ArgumentParser(description="Analyze Mario DDQN training logs.")
    parser.add_argument('log_file_path', type=str, help="Path to the log file to be analyzed.")
    parser.add_argument('--smoothing_window', type=int, default=0, help="Number of episodes for rolling average smoothing.")
    args = parser.parse_args()

    analyzer = LogAnalyzer(args.log_file_path)
    analyzer.analyze(smoothing_window=args.smoothing_window)

if __name__ == "__main__":
    main()
