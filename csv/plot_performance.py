import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv_files(directory='.', file_name=""):
    """
    Find and return a list of all CSV files in the specified directory.
    By default, the current directory is used.
    """
    # Construct a file pattern
    pattern = os.path.join(directory, f"{file_name}.csv")
    # Use glob to retrieve matching files
    csv_files = glob.glob(pattern)
    return csv_files


def load_dataframe(csv_files, dataframes=None, file_name=""):
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Verify that the necessary columns are present.
            if 'episode' in df.columns and 'accumulated_path_length' in df.columns:
                # Use the file name as the label for later plotting.
                dataframes[file_name] = df
            else:
                print(f"Warning: {file} does not contain required columns.")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return dataframes


def plot_performance(dataframes):
    """
    Create a single plot with each DataFrame's accumulated path length data.
    Each CSV file is plotted as a separate line, with the last one highlighted.
    Uses manually assigned colors for consistency.
    """
    plt.figure(figsize=(10, 6))

    # Manually define a list of colors (same as before)
    manual_colors = ['#8c564b', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#1f77b4', '#e377c2', '#1f77b4']

    items = list(dataframes.items())
    for i, (label, df) in enumerate(items):
        color = manual_colors[i % len(manual_colors)]
        is_last = (i == len(items) - 1)

        plt.plot(
            df['episode'],
            df['accumulated_path_length'],
            label=label,
            color=color,
            linewidth=4 if is_last else 1.5
        )

    plt.xlabel('Episode')
    plt.ylabel('Accumulated Path Length')
    plt.title('Accumulated Path Length per Episode Across 100 Episodes')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_performance_summary(dataframes):
    """
    Create a bar plot showing the mean accumulated path length across all episodes
    for each method (DataFrame), with standard deviation as error bars.
    Each bar uses a manually specified color.
    """
    labels = []
    means = []
    stds = []

    for label, df in dataframes.items():
        mean_val = df['path_length'].mean()
        std_val = df['path_length'].std()

        labels.append(label)
        means.append(mean_val)
        stds.append(std_val)

    x = np.arange(len(labels))

    # Manually define a list of colors (extend if you have more methods)
    manual_colors = ['#8c564b', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#1f77b4', '#e377c2', '#1f77b4']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, means, yerr=stds, capsize=10, alpha=0.8, color=manual_colors[:len(labels)])

    # Add shaded std dev areas
    for i in range(len(x)):
        plt.fill_between([x[i] - 0.2, x[i] + 0.2],
                         [means[i] - stds[i]] * 2,
                         [means[i] + stds[i]] * 2,
                         color=manual_colors[i],
                         alpha=0.3)

    plt.xticks(x, labels)
    plt.ylabel('Mean Path Length Across 100 Episodes')
    plt.title('Comparison of Methods: Mean Path Length with Std Dev')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # You can specify any directory where your CSV files are stored. Here we use the current directory.
    directory = '.'
    dataframes = {}
    csv_files = load_csv_files(directory, file_name="hybrid_data_075_025")
    dataframes = load_dataframe(csv_files, dataframes,
                                file_name="Hybrid\nIG: 0.75\nClosest: 0.25")
    csv_files = load_csv_files(directory, file_name="hybrid_data_05_05")
    dataframes = load_dataframe(csv_files, dataframes, file_name="Hybrid\nIG: 0.5\nClosest: 0.5")
    csv_files = load_csv_files(directory, file_name="hybrid_data_025_075")
    dataframes = load_dataframe(csv_files, dataframes,
                                file_name="Hybrid\nIG: 0.25\nClosest: 0.75")
    csv_files = load_csv_files(directory, file_name="tare_local")
    dataframes = load_dataframe(csv_files, dataframes, file_name="Tare Local")
    csv_files = load_csv_files(directory, file_name="closest_frontier")
    dataframes = load_dataframe(csv_files, dataframes, file_name="Closest Frontier")
    csv_files = load_csv_files(directory, file_name="Ours")
    dataframes = load_dataframe(csv_files, dataframes, file_name="Ours")

    if dataframes:  # Only plot if we have valid DataFrames
        plot_performance_summary(dataframes)
    else:
        print("No valid CSV files to plot.")
