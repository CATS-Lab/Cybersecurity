import numpy as np

from parameters import *
import pandas as pd


def generate_quadratic_trajectory():
    # 定义参数
    time_step = 0.1  # 时间步长
    a = -5 / 12  # 二次项系数
    b = 5  # 一次项系数
    c = 17.5  # 常数项

    time_index = np.arange(0, T/10, time_step)

    # 生成速度值
    speed = a * (time_index)**2 + b * (time_index) + c
    speed[time_index > 12] = a * 12 ** 2 + b * 12 + c
    acceleration = np.gradient(speed, time_step)

    # 初始化位移数组
    position = np.zeros_like(time_index)

    # 初始化位移的初值
    position[0] = 0  # 初始位移

    # 根据速度计算位移
    for step in range(1, len(time_index)):
        position[step] = position[step - 1] + (speed[step - 1] + speed[step]) / 2 * time_step

    # 创建 DataFrame 保存数据
    df = pd.DataFrame({
        'Trajectory_ID': 0,
        'Time_Index': time_index,
        'Pos_LV': position,
        'Speed_LV': speed,
        'Acc_LV': acceleration
    })

    df.to_csv('./trajectories/quadratic_trajectory.csv', index=False)


def generate_sin_trajectory():
    # 6， 0.1，3， 0.2
    # 定义参数
    average_speed = 25
    amplitude = 6
    frequency = 0.1  # 频率，您可以根据需要调整
    time_duration = T / 10  # 总时间持续时间
    time_step = 0.1  # 时间步长

    time_index = np.arange(0, time_duration, time_step)

    # 生成速度值
    speed = average_speed + amplitude * np.sin(2 * np.pi * frequency * time_index)
    acceleration = np.gradient(speed, time_step)

    # 初始化速度和位移数组
    velocity = np.zeros_like(acceleration)
    position = np.zeros_like(acceleration)

    # 初始化速度和位移的初值
    velocity[0] = average_speed  # 初始速度
    position[0] = 0  # 初始位移

    # 根据加速度计算速度和位移
    for step in range(1, len(time_index)):
        velocity[step] = velocity[step - 1] + acceleration[step - 1] * time_step
        position[step] = position[step - 1] + (velocity[step - 1] + velocity[step]) / 2 * time_step

    # 创建 DataFrame 保存数据
    df = pd.DataFrame({
        'Trajectory_ID': 0,
        'Time_Index': time_index,
        'Pos_LV': position,
        'Speed_LV': velocity,
        'Acc_LV': acceleration
    })

    df.to_csv('./trajectories/sin_trajectory_long2.csv', index=False)


def generate_linear_trajectory():
    time_duration = T / 10  # 总时间持续时间
    time_step = 0.1  # 时间步长
    average_speed = 25

    time_index = np.arange(0, time_duration, time_step)

    acceleration = np.zeros(T)
    acceleration[11:31] = -5
    acceleration[31:51] = 5
    acceleration[51:71] = -5
    acceleration[71:91] = 5
    acceleration[91:111] = -5
    acceleration[111:131] = 5

    # 初始化速度和位移数组
    velocity = np.zeros_like(acceleration)
    position = np.zeros_like(acceleration)

    # 初始化速度和位移的初值
    velocity[0] = average_speed  # 初始速度
    position[0] = 0  # 初始位移

    # 根据加速度计算速度和位移
    for step in range(1, len(time_index)):
        velocity[step] = velocity[step - 1] + acceleration[step - 1] * time_step
        position[step] = position[step - 1] + (velocity[step - 1] + velocity[step]) / 2 * time_step

    # 创建 DataFrame 保存数据
    df = pd.DataFrame({
        'Trajectory_ID': 0,
        'Time_Index': time_index,
        'Pos_LV': position,
        'Speed_LV': velocity,
        'Acc_LV': acceleration
    })

    df.to_csv('./trajectories/affine_trajectory.csv', index=False)

# generate_quadratic_trajectory()
generate_sin_trajectory()
# generate_linear_trajectory()

