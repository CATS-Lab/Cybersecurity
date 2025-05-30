import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model, GRB

# ====== Simulation and Model Parameters ======
dt = 0.1       # time step [s]
Nd = 3         # detection delay steps
Np = 25        # prediction horizon
Ns = 20        # number of future steps considered in safety headway
Nc = 0         # communication delay
T = 150        # simulation time steps
M = 100000     # big-M constant

# ====== Trajectory Optimization Weights and Bounds ======
w = [1, 0.1, 1]                # weights for objective function terms
v_range = [0, 50]             # velocity range [m/s]
a_range = [-5, 5]             # acceleration range [m/s^2]
eta1 = 3                      # hyperparameter (possibly for constraints)
eta2 = 2                      # hyperparameter
g_min = 2                     # minimum safety headway [m]


def read_trajectory(file, traj_id):
    """Load and preprocess a trajectory from file based on ID."""
    df = pd.read_csv(file)
    traj_data = df[df['Trajectory_ID'] == traj_id]
    traj_data_filtered = traj_data[traj_data['Time_Index'] < 150].head(T)

    p_l = traj_data_filtered['Pos_LV'].to_numpy()
    v_l = traj_data_filtered['Speed_LV'].to_numpy()
    a_l = traj_data_filtered['Acc_LV'].to_numpy()

    # Recalculate trajectory by integration (position from velocity, velocity from acceleration)
    for step in range(1, T):
        v_l[step] = v_l[step - 1] + a_l[step - 1] * dt
        p_l[step] = p_l[step - 1] + (v_l[step - 1] + v_l[step]) / 2 * dt

    # Initial state of following vehicle
    a_f_init = 0
    v_f_init = v_l[Nd]
    p_f_init = 0

    # Adjust position to satisfy initial headway
    headway = connected_safety_headway(v_l[Nd + Ns], v_l[Nd], p_l[Nd], p_l[Nd + Ns])
    p_l = p_l - p_l[Nd] + headway

    return a_l, v_l, p_l, a_f_init, v_f_init, p_f_init


def connected_safety_headway(vhl, vf, phl_1, phl_2):
    """Solve MILP to compute safety headway between leader and follower."""
    model = Model("Safety headway")
    model.setParam('OutputFlag', False)

    # Variables
    am = model.addVar(lb=a_range[0], ub=0, name="a")  # assumed max deceleration
    q = model.addVars(3, vtype=GRB.BINARY, name="q")  # mode selection
    h = model.addVar(lb=g_min, ub=GRB.INFINITY, name="h")  # headway

    # Constraints for different modes
    model.addConstr(M * (1 - q[0]) >= vhl - vf - Ns * dt * am + Nd * dt * a_range[1], name="q1")
    model.addQConstr(
        h + M * (1 - q[0]) >= -(vf + Ns * dt * am) ** 2 / (2 * a_range[0]) + (Nd + Ns) * dt * vf +
        (Nd + Ns / 2) * Ns * dt**2 * am - phl_2 + phl_1, name="headway1")

    model.addConstr(M * (1 - q[1]) >= vf + Ns * dt * am - Nd * dt * a_range[1] - vhl, name="q2_1")
    model.addConstr(M * (1 - q[1]) >= vhl - vf - Ns * dt * am, name="q2_2")
    model.addQConstr(
        h + M * (1 - q[1]) >= -(vhl + Nd * dt * a_range[1]) ** 2 / (2 * a_range[0]) + (Nd + Ns) * dt * vf
        - (vhl - vf - Ns * dt * am) ** 2 / (2 * a_range[1]) +
        (Nd + Ns / 2) * Ns * dt**2 * am + 0.5 * (Nd * dt) ** 2 * a_range[1] - phl_2 + phl_1,
        name="headway2")

    model.addConstr(M * (1 - q[2]) >= vf - Ns * dt * (-am) - vhl, name="q3")
    model.addQConstr(
        h + M * (1 - q[2]) >= -(vf + Ns * dt * am + Nd * dt * a_range[1]) ** 2 / (2 * a_range[0]) +
        (Nd + Ns) * dt * vf + (Nd + Ns / 2) * Ns * dt**2 * am +
        0.5 * (Nd * dt) ** 2 * a_range[1] - phl_2 + phl_1,
        name="headway3")

    model.addConstr(q[0] + q[1] + q[2] == 1, name="q_sum")  # only one mode is active

    model.setObjective(h - g_min, GRB.MINIMIZE)
    model.optimize()

    dm_opt = vhl  # default fallback
    if model.Status == GRB.OPTIMAL:
        dm_opt = h.X

    return dm_opt


def read_results(file):
    """Read optimized attack acceleration sequence from file."""
    df = pd.read_csv(file)
    return Nd + Np, df['Acc_attack'].iloc[Nd + Np:]
