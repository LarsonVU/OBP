import numpy as np
import pandas as pd
from gurobipy import Model, GRB

def readInput(excel_file_path):
    '''
    This function reads data from an excel file and returns it

    Input:
    - excel_file_path -> path to the excel file containing the data in the following order:
        - job_id
        - release_date
        - due_date
        - weight
        - st_1 <-> st_m (with m machines)

    Output:
    - data -> a pandas dataframe containing the data from the excel file
    '''

    data = pd.read_excel(excel_file_path)
    return data

def runAlgorithm(data, max_runtime):
    '''
    An ILP algorithm that uses gurobi to solve the permutation flowshop scheduling
    problem with release dates and total weighted tardiness as a objective function

    Input:
    - data -> a pandas dataframe containing the following column (in order):
        - job_id
        - release_date
        - due_date
        - weight
        - st_1 <-> st_m (with m machines)

    Output:
    - schedule -> the final schedule that the algorithm decided on
    - score -> the score of the schedule
    - runtime -> the runtime of the algorithm
    '''

    # Read the data from the dataframe
    job_ids = np.array(data.job_id)
    release_dates = np.array(data.release_date)
    due_dates = np.array(data.due_date)
    weights = np.array(data.weight)
    processing_times = np.array(data.iloc[:, 4:])

    # Get the number of jobs and number of machines
    num_jobs = len(job_ids)
    num_machines = processing_times.shape[1]

    # Create an array containing an index for all of the machines 
    machines = np.arange(1, num_machines + 1)

    # Initialize the model
    model = Model("TaskScheduling")

    # Initialize variables that indicate the completion time of a job j on machine k
    C = model.addVars(job_ids, machines, lb=0, vtype=GRB.CONTINUOUS, name="CompletionTime")

    # Initialize variables that indicate the tardiness of job j
    T = model.addVars(job_ids, lb=0, vtype=GRB.CONTINUOUS, name="Tardiness")

    # Initialize machine-specific order variables: x[i, j, k] -> whether task i precedes task j on machine k
    x = model.addVars(job_ids, job_ids, machines, vtype=GRB.BINARY, name="Order")

    # Set the objective to minimize the total weighted tardiness
    model.setObjective(sum(weights[j - 1] * T[j] for j in job_ids), GRB.MINIMIZE)

    # Add constraints for release dates
    model.addConstrs((C[j, 1] >= release_dates[j - 1] + processing_times[j - 1, 0] for j in job_ids), name="ReleaseDateConstr")

    # Add constraints for machine precedence (task j on machine k starts after machine k-1 finishes processing it)
    model.addConstrs((C[j, k] >= C[j, k - 1] + processing_times[j - 1, k - 1]
                    for j in job_ids for k in machines[1:]), name="MachinePrecedenceConstr")

    # Add constraints for task ordering on each machine
    M = np.sum(processing_times) + np.max(release_dates)  # Large constant
    model.addConstrs((
        C[j, k] >= C[i, k] + processing_times[j - 1, k - 1] * x[i, j, k] - M * (1 - x[i, j, k])
        for i in job_ids for j in job_ids for k in machines if i != j
    ), name="TaskOrderConstr")

    # Add constraints to ensure mutual exclusion of order variables on each machine
    model.addConstrs((x[i, j, k] + x[j, i, k] == 1 for i in job_ids for j in job_ids for k in machines if i != j),
                    name="MutualExclusion")

    # Add constraints to calculate tardiness
    model.addConstrs((T[j] >= C[j, num_machines] - due_dates[j - 1] for j in job_ids), name="TardinessConstr")

    # Add constraints to ensure that no task precedes itself on any machine
    model.addConstrs((x[j, j, k] == 0 for j in job_ids for k in machines), name="NoSelfPrecedenceConstr")

    # Set a time limit for the solver
    model.Params.TimeLimit = max_runtime

    # Solve the model
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        columns = ['Job ID']
        for i in range(len(machines)):
            columns += [f'Start time machine {i+1}', f'Completion time machine {i+1}']
        schedule = pd.DataFrame(columns=columns)

        for j in job_ids:
            job_schedule = [j]
            for k in machines:
                job_schedule += [C[j, k].X - processing_times[j - 1, k-1], C[j, k].X]
            schedule.loc[len(schedule)] = job_schedule

        score = model.objVal
        print("Schedule:\n", schedule)
        print("Score:", score)
        print("Runtime:", model.Runtime)
    else:
        print("No feasible solution found.")


# Example usage
if __name__ == "__main__":
    data = readInput('data/job_data3.xlsx')
    MAX_RUNTIME = 10
    schedule, score, runtime = runAlgorithm(data, MAX_RUNTIME)
    print("Schedule:\n", schedule)
    print("Score:", score)
    print("Runtime:", runtime)