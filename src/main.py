from upper_level import upper_level, upper_level_GA
from plot import *
from simulator import CFSimulator

# Loop through trajectory indices 1 to 8
for trj in range(1, 9):
    # Load reference trajectory
    if trj <= 4:
        # Use synthetic sine-based trajectories
        ref_trajectory = read_trajectory('./trajectories/sin_trajectory.csv', trj)
    else:
        # Use real-world trajectories extracted from Ultra-AV data
        ref_trajectory = read_trajectory('./trajectories/extracted_trajectory.csv', trj)

    # Solve the upper-level optimization problem using Genetic Algorithm (GA)
    solution, obj = upper_level_GA(ref_trajectory)
    # Alternatively, you can use a different solver:
    # solution, obj = upper_level(ref_trajectory)

    # Extract attack time and attack acceleration sequence from the solution
    attack_time = int(solution[0])
    attack_acc = solution[1:]
    # Load attack parameters from a result file
    # attack_time, attack_acc = read_results('./results/' + str(trj) + '_result.csv')

    # Initialize the simulator with attack parameters and trajectory
    simulator = CFSimulator(attack_time, attack_acc, ref_trajectory, trj)
    simulator.simulate()
    simulator.print_trajectory('./results/' + str(trj) + '_result.csv')

    # Visualization and analysis
    # draw_trajectory(str(trj))
    # draw_multiple_trajectory_compare_connected(str(trj))
    # draw_multiple_trajectory(str(trj))
