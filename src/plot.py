import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from parameters import *
import seaborn as sns
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Times New Roman'

# Plot single trajectory: position and speed
def draw_trajectory(trj, if_crash=False):
    df = pd.read_csv('./results/' + trj + '_result.csv')
    data = df[Nd + 1: -Np - 1]

    time_index = data['Time_Index']
    attack_index = ((data['Pos_attack'] - data['Pos_LV']).abs() > 1e-3).idxmax() - 1
    crash_index = (data['Pos_FV'] > data['Pos_LV']).idxmax()

    # Position plot
    plt.figure(figsize=(12, 10))
    plt.plot(time_index, data['Pos_LV'], color='red', label='LV', linewidth=5)
    plt.plot(time_index[attack_index - Nd:], data['Pos_attack'][attack_index - Nd:],
             color='gray', label='False Data', linestyle='--', linewidth=5)
    if if_crash:
        plt.plot(time_index[:crash_index - Nd], data['Pos_FV'][:crash_index - Nd],
                 color='blue', label='FV', linewidth=5)
        plt.scatter(time_index[crash_index], data['Pos_FV'][crash_index],
                    color='#DAA520', s=250, label='Crash Point', zorder=2)
    else:
        plt.plot(time_index, data['Pos_FV'], color='blue', label='FV', linewidth=5)
    plt.scatter(time_index[attack_index], data['Pos_attack'][attack_index],
                color='black', s=250, label='Attack Point', zorder=2)
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Position (m)', fontsize=60)
    plt.ylim(0, 400)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig('./results/' + trj + '_x.png')
    plt.close()

    # Speed plot
    plt.figure(figsize=(12, 10))
    plt.plot(time_index, data['Speed_LV'], color='red', label='LV', linewidth=5)
    plt.plot(time_index[attack_index - Nd:], data['Speed_attack'][attack_index - Nd:],
             color='gray', label='False Data', linestyle='--', linewidth=5)
    if if_crash:
        plt.plot(time_index[:crash_index - Nd], data['Speed_FV'][:crash_index - Nd],
                 color='blue', label='FV', linewidth=5)
        plt.scatter(time_index[crash_index], data['Speed_FV'][crash_index],
                    color='#DAA520', s=250, label='Crash Point', zorder=2)
    else:
        plt.plot(time_index, data['Speed_FV'], color='blue', label='FV', linewidth=5)
    plt.scatter(time_index[attack_index], data['Speed_attack'][attack_index],
                color='black', s=250, label='Attack Point', zorder=2)
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Speed (m/s)', fontsize=60)
    plt.ylim(5, 45)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig('./results/' + trj + '_v.png')
    plt.close()


# Compare with and without defense strategy
def draw_multiple_trajectory(trj):
    df = pd.read_csv('./results/' + trj + '_result.csv')
    data = df[Nd + 1: -Np - 1]
    df_new = pd.read_csv('./results/' + trj + '_result_s.csv')
    data_new = df_new[Nd + 1: -Np - 1]

    time_index = data['Time_Index']
    attack_index = ((data['Pos_attack'] - data['Pos_LV']).abs() > 1e-2).idxmax() - 1
    crash_index = (data['Pos_FV'] > data['Pos_LV']).idxmax()

    # Position plot
    plt.figure(figsize=(12, 10))
    plt.plot(time_index, data['Pos_LV'], color='red', label='LV', linewidth=5)
    plt.plot(time_index[attack_index - Nd:], data['Pos_attack'][attack_index - Nd:],
             color='gray', label='False Data', linestyle='--', linewidth=5)
    plt.plot(time_index[:crash_index - Nd], data['Pos_FV'][:crash_index - Nd],
             color='green', label='FV without strategy', linewidth=5)
    plt.plot(time_index, data_new['Pos_FV'], color='blue', label='FV with strategy', linewidth=5)
    plt.scatter(time_index[crash_index], data['Pos_FV'][crash_index],
                color='#DAA520', s=250, label='Crash Point', zorder=2)
    plt.scatter(time_index[attack_index], data['Pos_attack'][attack_index],
                color='black', s=250, label='Attack Point', zorder=2)
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Position (m)', fontsize=60)
    plt.ylim(0, 400)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig('./results/' + trj + '_x.png')
    plt.close()

    # Speed plot
    plt.figure(figsize=(12, 10))
    plt.plot(time_index, data['Speed_LV'], color='red', linewidth=5)
    plt.plot(time_index[attack_index - Nd:], data['Speed_attack'][attack_index - Nd:],
             color='gray', linestyle='--', linewidth=5)
    plt.plot(time_index[:crash_index - Nd], data['Speed_FV'][:crash_index - Nd],
             color='green', linewidth=5)
    plt.plot(time_index, data_new['Speed_FV'], color='blue', linewidth=5)
    plt.scatter(time_index[crash_index], data['Speed_FV'][crash_index],
                color='#DAA520', s=250, zorder=2)
    plt.scatter(time_index[attack_index], data['Speed_attack'][attack_index],
                color='black', s=250, zorder=2)
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Speed (m/s)', fontsize=60)
    plt.ylim(0, 50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig('./results/' + trj + '_v.png')
    plt.close()


# Compare connected vs non-connected vehicles
def draw_multiple_trajectory_compare_connected(trj):
    df = pd.read_csv('./results/' + trj + '_result.csv')
    data = df[Nd + 1: -Np - 1]
    df_new = pd.read_csv('./results/' + trj + '_result_n.csv')
    data_new = df_new[Nd + 1: -Np - 1]

    time_index = data['Time_Index']

    plt.figure(figsize=(12, 10))
    plt.plot(time_index, data['Pos_LV'], color='red', label='LV', linewidth=5)
    plt.plot(time_index, data['Pos_FV'], color='blue', label='FV-connected', linewidth=5)
    plt.plot(time_index, data_new['Pos_FV'], color='green', label='FV-non-connected', linewidth=5)
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Position (m)', fontsize=60)
    plt.ylim(0, 300)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig('./results/' + trj + '_x.png')
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.plot(time_index, data['Speed_LV'], color='red', linewidth=5)
    plt.plot(time_index, data['Speed_FV'], color='blue', linewidth=5)
    plt.plot(time_index, data_new['Speed_FV'], color='green', linewidth=5)
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Speed (m/s)', fontsize=60)
    plt.ylim(10, 33)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig('./results/' + trj + '_v.png')
    plt.close()


# Plot convergence curves for GA optimization
def converge():
    generations = np.arange(1, 21)
    f_avg = [...]  # average objective values per generation
    f_min = [...]  # best objective values per generation

    plt.figure(figsize=(8, 6))
    plt.plot(generations, f_avg, label='Objective value')
    plt.xlabel('Iterations', fontsize=30)
    plt.ylabel('Average objective values', fontsize=30)
    plt.xticks(np.arange(1, 21, 1), fontsize=20)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('./results/converge_avg.png')

    plt.figure(figsize=(8, 6))
    plt.plot(generations, f_min, label='Objective value')
    plt.xlabel('Iterations', fontsize=30)
    plt.ylabel('Optimal objective values', fontsize=30)
    plt.xticks(np.arange(1, 21, 1), fontsize=20)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig('./results/converge_opt.png')


# Plot headway vs time for each trajectory
def headway():
    df = pd.read_excel('./results/summary.xlsx')
    trajectory_ids = df['Trajectory_ID'].unique()

    for trajectory_id in trajectory_ids:
        subset_df = df[df['Trajectory_ID'] == trajectory_id]
        time_index = [t / 10 for t in range(len(subset_df))]

        plt.figure(figsize=(12, 10))
        plt.plot(time_index, subset_df['safety'], label='Safety headway', linewidth=5, linestyle='--')
        plt.plot(time_index, subset_df['headway'], label='Real headway', linewidth=5)
        plt.xlabel('Time (s)', fontsize=60)
        plt.ylabel('Headway (m)', fontsize=60)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.tight_layout()
        plt.savefig(f'./results/headway_{trajectory_id}.png')


# Plot runtime statistics
def runtime():
    df = pd.read_csv('./results/runtime.csv')
    all_times = df.melt(value_name='Time')['Time']
    color = (135 / 255, 206 / 255, 235 / 255)  # skyblue

    # Violin plot
    plt.figure(figsize=(8, 8))
    sns.violinplot(y=all_times, inner="box", color=color)
    plt.ylabel('Time (s)', fontsize=30)
    plt.xticks([])
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig('./results/runtime_violin.png')

    # Histogram and density plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    sns.histplot(all_times, bins=30, kde=False, color=color, ax=ax1)
    ax1.set_xlabel('Time (s)', fontsize=35)
    ax1.set_ylabel('Frequency', fontsize=35)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)

    ax2 = ax1.twinx()
    sns.kdeplot(all_times, ax=ax2, color="red")
    ax2.set_ylabel('Probability Density', fontsize=35)
    ax2.tick_params(axis='y', labelsize=20)

    fig.tight_layout()
    plt.savefig('./results/runtime_histogram.png')
