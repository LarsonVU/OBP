from gurobipy import Model, GRB

# Input data
tasks = [1, 2, 3]  # Task IDs
release_dates = {1: 0, 2: 3, 3: 5}
due_dates = {1: 10, 2: 15, 3: 20}
service_times = {1: 3, 2: 5, 3: 2}
weights = {1: 1, 2: 2, 3: 1}

# Create model
model = Model("TaskScheduling")

# Variables
S = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="Start")
C = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="Completion")
T = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="Tardiness")
x = model.addVars(tasks, tasks, vtype=GRB.BINARY, name="Order")

# Objective: Minimize weighted tardiness
model.setObjective(sum(weights[i] * T[i] for i in tasks), GRB.MINIMIZE)

# Constraints
for i in tasks:
    model.addConstr(S[i] >= release_dates[i])  # Release date constraint
    model.addConstr(C[i] == S[i] + service_times[i])  # Completion time constraint
    model.addConstr(T[i] >= C[i] - due_dates[i])  # Tardiness definition
    model.addConstr(T[i] >= 0)  # Non-negative tardiness

for i in tasks:
    for j in tasks:
        if i != j:
            model.addConstr(S[j] >= C[i] - (1 - x[i, j]) * 1e6)  # Non-overlapping constraint
            model.addConstr(S[i] >= C[j] - x[i, j] * 1e6)  # Reverse order constraint
            model.addConstr(x[i, j] + x[j, i] == 1)  # Mutual exclusivity

# Solve
model.optimize()

# Output results
for i in tasks:
    print(f"Task {i}: Start = {S[i].X}, Completion = {C[i].X}, Tardiness = {T[i].X}")
