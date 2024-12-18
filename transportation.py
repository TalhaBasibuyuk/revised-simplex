import pulp
import numpy as np

# A data class to hold the instances of the Transportation Problem
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

        # Following loop makes sure that sum of demand nodes and supply nodes match to avoid adding dummy variables
        # It distributes the difference among other nodes until the difference becomes 0
        difference = sum(supply_nodes) - sum(demand_nodes)  # difference between supply and demand
        i = 0
        while difference != 0:
            if difference > 0:
                amount_to_change = min(supply_nodes[i] - 1, difference)
                difference -= amount_to_change
                supply_nodes[i] -= amount_to_change
            else:
                amount_to_change = min(max_demand_supply - supply_nodes[i], -difference)
                difference += amount_to_change
                supply_nodes[i] += amount_to_change
            i += 1

        # Build the cost matrix using random integers.
        cost_matrix = np.random.randint(low=1, high=max_cost + 1, size=(supply_nodes_amount, demand_nodes_amount))

        # Return the generated transportation instance
        return TransportationInstance(supply_nodes, demand_nodes, cost_matrix)


# Solves the given transportation problem instance using pulp library.
def solve_transportation_problem(transportation_instance):

    # Get the problem parameters
    supply_nodes = transportation_instance.supply_nodes
    demand_nodes = transportation_instance.demand_nodes
    cost_matrix = transportation_instance.cost_matrix

    # Formulate the problem
    problem = pulp.LpProblem("transportation_problem", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(len(demand_nodes))] for i in
         range(len(supply_nodes))]
    # Costs
    problem += pulp.lpSum(
        cost_matrix[i][j] * x[i][j] for j in range(len(demand_nodes)) for i in range(len(supply_nodes)))
    # Supply constraints
    for i in range(len(supply_nodes)):
        problem += pulp.lpSum(x[i][j] for j in range(len(demand_nodes))) == supply_nodes[i]
    # Demand constraints
    for j in range(len(demand_nodes)):
        problem += pulp.lpSum(x[i][j] for i in range(len(supply_nodes))) == demand_nodes[j]

    problem.solve()
    solution = [[x[i][j].varValue for j in range(len(demand_nodes))] for i in range(len(supply_nodes))]
    optimal_value = problem.objective.value()

    return solution, optimal_value


# Solves the given LP problem with Big-M and Revised Simplex Method
def revised_simplex_solver(cost_vector, constraint_matrix, b, lp_type = 1):
    # Determine the signs of the M and the cost matrix depending on problem type,
    # Max problem
    sign = 1
    # Update the sign if it's a min problem
    if (lp_type == 0):
        for i in range(len(cost_vector)):
            sign = -1
            cost_vector[i] = cost_vector[i] * (-1)

    # Determine a number to model the huge penalty for artificial variables
    M = 10**5 * abs(max(b))

    # Assume all constraints take artificial variables and apply big_m
    n = len(constraint_matrix)
    I_n = np.eye(n)
    augmented_constraint_matrix = np.concatenate((constraint_matrix, I_n), axis=1)
    canonical_cost_vector = cost_vector + [-1 * M for _ in range(n)]

    # Holds the amount added to objective value during row operations, this amount is subtracted at the end to get
    # the actual objective value
    added_z = 0

    # Make the objective row canonical
    for i in range(n):
        canonical_cost_vector += M * augmented_constraint_matrix[i]
        added_z += b[i] * M

    return sign * (revised_simplex(canonical_cost_vector, augmented_constraint_matrix, b)[0] - added_z)


def revised_simplex(cost_vector, constraint_matrix, b):

    # Following are the structures we need to use in revised simplex iterations
    constraint_matrix = np.array(constraint_matrix)
    number_of_basic_variables, number_of_variables = constraint_matrix.shape
    number_of_nonbasic_variables = number_of_variables - number_of_basic_variables

    # x_b, x_n
    basic_indices = list(range(number_of_nonbasic_variables, number_of_variables))
    nonbasic_indices = list(range(number_of_nonbasic_variables))

    # c_b, c_n
    basic_costs = cost_vector[basic_indices[0]:basic_indices[-1] + 1]
    nonbasic_costs = cost_vector[nonbasic_indices[0]:nonbasic_indices[-1] + 1]

    # We divided the constraint_matrix into following matrices depending on whether the variable is basic or nonbasic.
    basic_variables_matrix = np.zeros((number_of_basic_variables, number_of_basic_variables))
    nonbasic_variables_matrix = np.zeros((number_of_basic_variables, number_of_nonbasic_variables))

    # Here we fill the matrices with the data in constraint_matrix
    for column in range(number_of_variables):
        for row in range(number_of_basic_variables):
            if column < basic_indices[0]:
                nonbasic_variables_matrix[row][column] = constraint_matrix[row][column]
            else:
                basic_variables_matrix[row][column - number_of_nonbasic_variables] = constraint_matrix[row][column]



    # Iterations of revised_simplex algorithm, lasts until optimality check fails or unboundedness or infeasibility
    while True:
        inverse_of_basic_variables_matrix = np.linalg.inv(basic_variables_matrix)

        # Here we check the Optimality to see if the iterations proceed.
        reduced_costs = basic_costs @ inverse_of_basic_variables_matrix @ nonbasic_variables_matrix - nonbasic_costs

        # Determine the entering variable
        min_cost = float("inf")
        entering_index = 0
        entering_variable = 0
        for i in range(len(reduced_costs)):
            if reduced_costs[i] < min_cost:
                min_cost = reduced_costs[i]
                entering_index = i
                entering_variable = nonbasic_indices[entering_index]

        # The solution is optimal if there are no a reduced cost lest than zero, so we can return the solution.
        if min_cost >= 0:
            optimal_solution = inverse_of_basic_variables_matrix @ b
            optimal_value = basic_costs @ inverse_of_basic_variables_matrix @ b
            return optimal_value, optimal_solution, basic_variables_matrix, nonbasic_variables_matrix

        # We determine the leaving variable by applying the minimum ratio test to pivot_column
        pivot_column = inverse_of_basic_variables_matrix @ nonbasic_variables_matrix[:, entering_index]
        rhs_vector = inverse_of_basic_variables_matrix @ b
        min_ratio = float("inf")
        leaving_index = -1
        leaving_variable = -1

        for i in range(len(basic_indices)):
            # We can't apply MRT to negative and zero coefficients
            if pivot_column[i] <= 0:
                continue
            if rhs_vector[i] < 0:
                # Problem is infeasible if we get a negative RHS
                return "infeasible", "infeasible"

            current_ratio = rhs_vector[i] / pivot_column[i]
            if current_ratio < min_ratio:
                min_ratio = current_ratio
                leaving_index = i
                leaving_variable = basic_indices[leaving_index]

        # check if the problem is unbounded
        if leaving_index == -1:
            return "unbounded", "INF"

        # update matrices
        basic_indices[leaving_index] = entering_variable
        nonbasic_indices[entering_index] = leaving_variable

        # update c_b and c_n
        basic_costs[leaving_index], nonbasic_costs[entering_index] = nonbasic_costs[entering_index], basic_costs[leaving_index]

        # Update coefficient matrices
        temp_column = basic_variables_matrix[:, leaving_index].copy()
        basic_variables_matrix[:, leaving_index] = nonbasic_variables_matrix[:, entering_index]
        nonbasic_variables_matrix[:, entering_index] = temp_column


max_cost = 100
max_demand_supply = 100
for i in range(5):
    amount = np.random.randint(1, 15)
    prob = TransportationInstance.generate_transportation_instance(amount, amount, max_cost, max_demand_supply)
    #print(solve_transportation_problem(prob))
    #print(revised_simplex_solver([2, 3, 0, 0], [[1,2,1,0], [3, 4, 0, 1]], [10, 20]))
    #print(revised_simplex_solver([1, 1, 1, 1, 1, 0, 0, 0], [[6, 6, 6, 4, 4, 1, 0, 0] , [2, 5, 5, 3, 4, 0, 1, 0], [5, 4, 4, 3, 2, 0, 0, 1]], [87, 77, 57]))
    #print(revised_simplex_solver([3, 2, 1, 0, 0, 0], [[7, 9, 11, -1, 0, 0] , [4, 42, 21, 0, 1, 0], [10, 11, 17, 0, 0, -1]], [370, 4384, 1000]))

    print("here")
