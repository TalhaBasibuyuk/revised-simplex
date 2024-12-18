import pulp
import numpy as np


class TransportationInstance:

    def _init_(self, supply_nodes, demand_nodes, cost_matrix):
        self.supply_nodes = supply_nodes
        self.demand_nodes = demand_nodes
        self.cost_matrix = cost_matrix

    # Generates a feasible instance of the transportation problem.
    @staticmethod
    def generate_transportation_instance(supply_nodes_amount, demand_nodes_amount, max_cost, max_demand_supply):
        supply_nodes = np.random.randint(low=1, high=max_demand_supply + 1, size=supply_nodes_amount).tolist()  # a_i
        demand_nodes = np.random.randint(low=1, high=max_demand_supply + 1, size=demand_nodes_amount).tolist()  # b_j

        # make sure supply = demand to avoid dummy
        s = sum(supply_nodes) - sum(demand_nodes)  # aurplua/slack
        i = 0
        while s != 0:
            if s > 0:
                k = min(supply_nodes[i] - 1, s)
                s -= k
                supply_nodes[i] -= k
            else:
                k = min(max_demand_supply - supply_nodes[i], -s)
                s += k
                supply_nodes[i] += k
            i += 1

        # create cost matrix
        cost_matrix = np.random.randint(low=1, high=max_cost + 1, size=(supply_nodes_amount, demand_nodes_amount))
        return TransportationInstance(supply_nodes, demand_nodes, cost_matrix)


def solve_transportation_problem(transportation_instance):
    supply_nodes = transportation_instance.supply_nodes
    demand_nodes = transportation_instance.demand_nodes
    cost_matrix = transportation_instance.cost_matrix

    problem = pulp.LpProblem("transportation_problem", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(len(demand_nodes))] for i in
         range(len(supply_nodes))]
    problem += pulp.lpSum(
        cost_matrix[i][j] * x[i][j] for j in range(len(demand_nodes)) for i in range(len(supply_nodes)))  # obj fnc
    for i in range(len(supply_nodes)):  # constraints for supply&demand
        problem += pulp.lpSum(x[i][j] for j in range(len(demand_nodes))) == supply_nodes[i]
    for j in range(len(demand_nodes)):
        problem += pulp.lpSum(x[i][j] for i in range(len(supply_nodes))) == demand_nodes[j]

    problem.solve()
    solution = [[x[i][j].varValue for j in range(len(demand_nodes))] for i in range(len(supply_nodes))]
    optimal_value = problem.objective.value()

    return solution, optimal_value






def apply_big_m(cost_vector, constraint_matrix, b, lp_type = 1):
    sign = 1 if lp_type == 1 else -1
    M = 10**6
    # Assume all constraints take artificial var and apply big_m
    n = len(constraint_matrix)
    I_n = np.eye(n)
    augmented_constraint_matrix = np.concatenate((constraint_matrix, I_n), axis=1)
    canonical_cost_vector = cost_vector + [-M for _ in range(n)]
    print(canonical_cost_vector)
    added_z = 0

    # Make canonical
    for i in range(n):
        canonical_cost_vector += M * augmented_constraint_matrix[i]
        added_z += b[i] * M
        #canonical_cost_vector *= -1
    print(canonical_cost_vector)
    return revised_simplex(canonical_cost_vector, augmented_constraint_matrix, b, 1)[0] - added_z


def revised_simplex_solver2(cost_vector, constraint_matrix, b):
    phase1_return_tuple = phase_1(cost_vector, constraint_matrix, b)

    if phase1_return_tuple[0] != 0:
        print("Infeasible")
        return "Infeasible"
    basic, total = np.array(constraint_matrix).shape
    nonbasic = total - basic
    phase2_constraint_matrix = np.concatenate((phase1_return_tuple[3][:, 0:nonbasic], phase1_return_tuple[2]), axis=1)  # problem
    print("\n", phase2_constraint_matrix, "\n*")

    phase2_cost_vector = cost_vector + [0 for _ in range(len(constraint_matrix))]

    # make canonical




    return revised_simplex(phase2_cost_vector, phase2_constraint_matrix, phase1_return_tuple[1], 1)

def phase_1(cost_vector, constraint_matrix, b):
    # Assume all constraints take artificial var and apply two phase.
    n = len(constraint_matrix)
    I_n = np.eye(n)
    augmented_constraint_matrix = np.concatenate((constraint_matrix, I_n), axis=1)
    phase_1_cost_vector = [0 for _ in range(len(constraint_matrix[0]))] + [1 for _ in range(n)]

    print(constraint_matrix)
    print("*")
    print(augmented_constraint_matrix)
    print(phase_1_cost_vector)

    # Make Canonical
    for i in range(n):
        phase_1_cost_vector -= augmented_constraint_matrix[i]
    phase_1_cost_vector *= -1
    print(phase_1_cost_vector)
    return revised_simplex(phase_1_cost_vector, augmented_constraint_matrix, b, 1)


def revised_simplex(cost_vector, constraint_matrix, b, lp_type):

    sign = -1 if lp_type == 0 else 1
    # inits
    constraint_matrix = np.array(constraint_matrix)
    print(constraint_matrix.shape)
    number_of_basic_variables, number_of_variables = constraint_matrix.shape
    number_of_nonbasic_variables = number_of_variables - number_of_basic_variables
    # x_b, x_n
    basic_indices = list(range(number_of_nonbasic_variables, number_of_variables))
    nonbasic_indices = list(range(number_of_nonbasic_variables))

    # c_b, c_n
    basic_costs = cost_vector[basic_indices[0]:basic_indices[-1] + 1]
    nonbasic_costs = cost_vector[nonbasic_indices[0]:nonbasic_indices[-1] + 1]

    basic_variables_matrix = np.zeros((number_of_basic_variables, number_of_basic_variables))
    nonbasic_variables_matrix = np.zeros((number_of_basic_variables, number_of_nonbasic_variables))

    for column in range(number_of_variables):
        for row in range(number_of_basic_variables):
            if column < basic_indices[0]:
                nonbasic_variables_matrix[row][column] = constraint_matrix[row][column]
            else:
                basic_variables_matrix[row][column - number_of_nonbasic_variables] = constraint_matrix[row][column]



    # iterations
    while True:
        inverse_of_basic_variables_matrix = np.linalg.inv(basic_variables_matrix)
        # Here we multiply by -1 because transform max to min
        # opt check

        reduced_costs = sign * (basic_costs @ inverse_of_basic_variables_matrix @ nonbasic_variables_matrix - nonbasic_costs)
        min_cost = float("inf")
        entering_index = 0
        entering_variable = 0
        for i in range(len(reduced_costs)):
            # fingding entering
            if reduced_costs[i] < min_cost:
                min_cost = reduced_costs[i]
                entering_index = i
                entering_variable = nonbasic_indices[entering_index]
        if min_cost >= 0:
            optimal_solution = inverse_of_basic_variables_matrix @ b
            optimal_value = sign * (basic_costs @ inverse_of_basic_variables_matrix @ b)
            return optimal_value, optimal_solution, basic_variables_matrix, nonbasic_variables_matrix

        pivot_column1 = inverse_of_basic_variables_matrix @ nonbasic_variables_matrix[:, entering_index]
        print("Test: ", " ", pivot_column1)
        rhs_vector = inverse_of_basic_variables_matrix @ b
        min_ratio = float("inf")
        leaving_index = -1
        leaving_variable = -1
        # fingding leaving
        for i in range(len(basic_indices)):
            if pivot_column1[i] <= 0:
                continue
            if rhs_vector[i] < 0:
                return "infeasible", "infeasible"

            current_ratio = rhs_vector[i] / pivot_column1[i]
            if current_ratio < min_ratio:
                min_ratio = current_ratio
                leaving_index = i
                leaving_variable = basic_indices[leaving_index]

        # check if bounded
        if leaving_index == -1:
            return "unbounded", "INF"

        # update matrices
        basic_indices[leaving_index] = entering_variable
        nonbasic_indices[entering_index] = leaving_variable

        # update c_b and c_n
        basic_costs[leaving_index], nonbasic_costs[entering_index] = nonbasic_costs[entering_index], basic_costs[leaving_index]
        print("BEFORE:")
        print(basic_variables_matrix)
        print(nonbasic_variables_matrix)
        temp_column = basic_variables_matrix[:, leaving_index].copy()
        basic_variables_matrix[:, leaving_index] = nonbasic_variables_matrix[:, entering_index]
        nonbasic_variables_matrix[:, entering_index] = temp_column
        print("AFTER:")
        print(basic_variables_matrix)
        print(nonbasic_variables_matrix)
        print()


for i in range(1):
    prob = TransportationInstance.generate_transportation_instance(3, 3, 100, 100)
    #print(solve_transportation_problem(prob))
    #print(apply_big_m([2, 3, 0, 0], [[1,2,1,0], [3, 4, 0, 1]], [10, 20]))
    #print(apply_big_m([1, 1, 1, 1, 1, 0, 0, 0], [[6, 6, 6, 4, 4, 1, 0, 0] , [2, 5, 5, 3, 4, 0, 1, 0], [5, 4, 4, 3, 2, 0, 0, 1]], [87, 77, 57]))
    print(apply_big_m([3, 2, 1, 0, 0, 0], [[7, 9, 11, -1, 0, 0] , [4, 42, 21, 0, 1, 0], [10, 11, 17, 0, 0, -1]], [370, 4384, 1000]))

    print("here")
