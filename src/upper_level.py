import geatpy as ea
from parameters import *
from simulator import CFSimulator
import nevergrad as ng


class MyProblem(ea.Problem):
    """A class to define optimization problems for evolutionary algorithms."""

    def __init__(self, lb, ub, ref_trajectory):
        self.ref_trajectory = ref_trajectory
        M = 1  # Number of objectives
        maxOrMin = [1]  # 1 means minimization
        Dim = len(lb)  # Number of decision variables
        varTypes = [1] + [0] * (Dim - 1)  # 0 for continuous, 1 for integer (first one is attack time)
        lbin = [1] * Dim  # 1 means lower bound is included
        ubin = [1] * Dim  # 1 means upper bound is included
        ea.Problem.__init__(self, "", M, maxOrMin, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """Objective function to evaluate each individual."""
        x = pop.Phen
        results = []
        constraints = []

        for solution in x:
            attack_time = int(solution[0])        # x[0]: attack time
            attack_acc = solution[1:]             # x[1:]: attack acceleration sequence

            simulator = CFSimulator(attack_time, attack_acc, self.ref_trajectory)
            simulator.simulate()

            TTC = np.min(simulator.TTCs)
            v = simulator.vh_l

            results.append(TTC)
            constraints.append(np.concatenate((v_range[0] - v, v - v_range[1])))  # speed range constraint

        pop.CV = np.vstack(constraints)
        pop.ObjV = np.vstack(results)  # Assign objective values to the population


def upper_level_GA(ref_trajectory):
    """Run the upper-level optimization using genetic algorithm (GEATPy)."""
    a_l, v_l, p_l, a_f_init, v_f_init, p_f_init = ref_trajectory
    problem = MyProblem(
        np.concatenate((np.array([Nd + Np]), a_l[Nd + Np:])),
        [T - Np] + [a_range[1]] * (T - Nd - Np),
        ref_trajectory
    )
    Encoding = 'RI'
    NIND = 50  # Population size
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.soea_SEGA_templet(problem, population)
    myAlgorithm.MAXGEN = 10       # Max generations
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 1

    # Optional: use a predefined population
    # prophetChrom = np.zeros((NIND, T - Nd + 1))
    # prophetChrom[0:, 0] = Nd + 1
    # prophetPop = ea.Population(Encoding, Field, NIND, prophetChrom)
    # myAlgorithm.call_aimFunc(prophetPop)
    # BestIndi, population = myAlgorithm.run(prophetPop)

    BestIndi, population = myAlgorithm.run()

    print('Optimal objective value:', BestIndi.ObjV[0][0])
    print('Best solution:', BestIndi.Phen[0])

    return BestIndi.Phen[0], BestIndi.ObjV[0][0]


def objective_function_factory(ref_trajectory):
    """
    Wrap the CFSimulator into an objective function with named parameters.
    Input:
        - attack_time: attack start step
        - attack_acc: array of acceleration values
    Output:
        - Minimum TTC value (to be minimized)
    """
    def objective_function(*, attack_time, attack_acc):
        attack_time = int(attack_time)
        attack_acc = np.array(attack_acc)
        simulator = CFSimulator(attack_time, attack_acc, ref_trajectory)
        simulator.simulate()
        return np.min(simulator.TTCs)
    return objective_function


def create_instrumentation(lb, ub):
    """
    Create decision variable descriptors for Nevergrad optimization.
    Sets variable bounds and initial values.
    """
    attack_time = ng.p.Scalar()
    attack_time.value = (lb[0] + ub[0]) // 2
    attack_time = attack_time.set_bounds(lb[0], ub[0]).set_integer_casting()

    attack_acc = ng.p.Array(shape=(len(lb) - 1,))
    attack_acc.value = np.array(lb[1:])
    attack_acc = attack_acc.set_bounds(lb[1:], ub[1:])

    return ng.p.Instrumentation(
        attack_time=attack_time,
        attack_acc=attack_acc
    )


def upper_level(ref_trajectory, budget=200):
    """
    Test multiple optimization algorithms using Nevergrad.
    Inputs:
        - ref_trajectory: the reference trajectory from the lead vehicle
        - budget: maximum number of evaluations per algorithm
    """
    algorithms = ["OnePlusOne"]  # Alternative: ["PSO", "DE", ...]

    a_l, v_l, p_l, a_f_init, v_f_init, p_f_init = ref_trajectory

    # Construct bounds
    lb = np.concatenate((np.array([Nd + Np]), a_l[Nd + Np:]))
    ub = [T - Np] + [a_range[1]] * (T - Nd - Np)

    instrumentation = create_instrumentation(lb, ub)

    best_result = {"solution": None, "obj": float("inf")}
    for algo_name in algorithms:
        print(f"\nRunning algorithm: {algo_name}")
        optimizer = ng.optimizers.registry[algo_name](instrumentation, budget=budget)
        objective_function = objective_function_factory(ref_trajectory)
        recommendation = optimizer.minimize(objective_function)

        best_attack_time = recommendation.kwargs["attack_time"]
        best_attack_acc = recommendation.kwargs["attack_acc"]
        solution = np.concatenate(([best_attack_time], best_attack_acc))
        obj = recommendation.loss

        print(f"Best result for {algo_name}: {obj}")

        if obj is not None and obj < best_result["obj"]:
            best_result["solution"] = solution
            best_result["obj"] = obj

    return best_result["solution"], best_result["obj"]
