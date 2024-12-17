import pulp
import numpy as np


class TransportationInstance:

    def __init__(self, supply_nodes, demand_nodes, cost_matrix):
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




def revised_simplex_solver(cost_vector, constraint_matrix, b):
    phase1_return_tuple = phase_1(cost_vector, constraint_matrix, b)

    if phase1_return_tuple[0] != 0:
        print("Infeasible")
        return "Infeasible"

    phase2_constraint_matrix = np.concatenate((phase1_return_tuple[2], phase1_return_tuple[3]), axis=1)  # problem

    phase2_cost_vector = cost_vector + [0 for _ in range(len(constraint_matrix))]

    return revised_simplex(phase2_cost_vector, phase2_constraint_matrix, phase1_return_tuple[1])

def phase_1(cost_vector, constraint_matrix, b):
    # Assume all constraints take artificial var and apply two phase.
    n = len(constraint_matrix)
    I_n = np.eye(n)
    augmented_constraint_matrix = np.hstack((constraint_matrix, I_n))
    phase_1_cost_vector = [0 for _ in range(len(constraint_matrix[0]))] + [1 for _ in range(n)]

    return revised_simplex(phase_1_cost_vector, augmented_constraint_matrix, b)


def revised_simplex(cost_vector, constraint_matrix, b):
    # inits
    constraint_matrix = np.array(constraint_matrix)

    number_of_basic_variables, number_of_variables = constraint_matrix.shape
    number_of_nonbasic_variables = number_of_variables - number_of_basic_variables
    # x_b, x_n
    basic_indices = list(range(number_of_nonbasic_variables, number_of_variables))
    nonbasic_indices = list(range(number_of_nonbasic_variables))


    print(cost_vector)
    # c_b, c_n
    basic_costs = cost_vector[basic_indices[0]:basic_indices[-1] + 1]
    nonbasic_costs = cost_vector[nonbasic_indices[0]:nonbasic_indices[-1] + 1]
    print(basic_costs)
    print(nonbasic_costs)

    columns = constraint_matrix.T


    basic_columns = np.zeros((number_of_basic_variables, number_of_basic_variables))
    nonbasic_columns = np.zeros((number_of_nonbasic_variables, number_of_basic_variables))
    for i in range(number_of_variables):
        if i < basic_indices[0]:
            nonbasic_columns[i] = columns[i]
        else:
            basic_columns[i- number_of_nonbasic_variables] = columns[i]

    # iterations
    non_optimal = True
    while True:
        inverse_of_basic_columns = np.linalg.inv(basic_columns)
        # Here we multiply by -1 because transform max to min
        # opt check
        print(nonbasic_columns)
        print("*****")

        print(basic_costs)
        print("*****")
        print(inverse_of_basic_columns)
        print("*****")
        print(nonbasic_columns)
        temp_nonbasic = nonbasic_columns.T.reshape(2, 4)
        reduced_costs = (-1) * (basic_costs @ inverse_of_basic_columns @ temp_nonbasic - nonbasic_costs)
        min_cost = float("inf")
        entering_index = 0
        for i in range(len(reduced_costs)):
            # fingding entering
            if reduced_costs[i] < min_cost:
                min_cost = reduced_costs[i]
                entering_index = i
        if min_cost >= 0:
            optimal_solution = inverse_of_basic_columns @ b
            optimal_value = (-1) * (basic_costs @ inverse_of_basic_columns @ b)
            return optimal_value, optimal_solution, basic_columns, nonbasic_columns

        print(inverse_of_basic_columns, "a\n", nonbasic_columns[entering_index])
        pivot_column = inverse_of_basic_columns @ nonbasic_columns[entering_index]
        print("p:", pivot_column)
        rhs_vector = inverse_of_basic_columns @ b
        min_ratio = float("inf")
        leaving_index = -1
        # fingding leaving
        for i in range(len(basic_indices)):
            if pivot_column[i] == 0:
                continue
            if rhs_vector[i] < 0:
                return "infeasible", "infeasible"
            current_ratio = rhs_vector[i] / pivot_column[i]
            if current_ratio < min_ratio:
                min_ratio = current_ratio
                leaving_index = i

        # check if bounded
        if leaving_index == -1:
            return "unbounded", "INF"

        # update matrices
        #basic_indices.remove(leaving_index)
        #nonbasic_indices.remove(entering_index)
        #nonbasic_indices.append(leaving_index)
        # basic_indices.append(entering_index)

        basic_columns[leaving_index], nonbasic_columns[entering_index] = (nonbasic_columns[entering_index],
                                                                          basic_columns[leaving_index])



for i in range(1):
    prob = TransportationInstance.generate_transportation_instance(3, 3, 100, 100)
    #print(solve_transportation_problem(prob))
    print(revised_simplex_solver([-3, -4, 0, 0], [[1, 2, 1, 0], [1, 1, 0, 1]], [8, 6] ))


