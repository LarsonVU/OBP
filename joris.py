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
    An ILP algorithm that uses Gurobi to solve the scheduling problem with machine-specific task ordering.

    Input:
    - data -> a pandas dataframe containing the following columns:
        - job_id
        - release_date
        - due_date
        - weight
        - st_1 <-> st_m (with m machines)
    - max_runtime -> maximum runtime for the solver.

    Output:
    - machine_schedules -> a dictionary containing the schedule for each machine.
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

    # Initialize variables that indicate the start time of a job j on machine k
    S = model.addVars(job_ids, machines, lb=0, vtype=GRB.CONTINUOUS, name="StartTime")

    # Initialize variables that indicate if job i precedes job j on machine k
    x = model.addVars(job_ids, job_ids, machines, vtype=GRB.BINARY, name="Order")

    # Set the objective to minimize the total weighted tardiness
    T = model.addVars(job_ids, lb=0, vtype=GRB.CONTINUOUS, name="Tardiness")
    model.setObjective(
        sum(weights[j - 1] * T[j] for j in job_ids), 
        GRB.MINIMIZE
    )

    # Add constraints for release dates
    model.addConstrs((S[j, 1] >= release_dates[j - 1] for j in job_ids), name="ReleaseDateConstr")

    # Add constraints for processing times and machine precedence
    model.addConstrs((C[j, k] == S[j, k] + processing_times[j - 1, k - 1] for j in job_ids for k in machines), name="ProcessingTimeConstr")
    model.addConstrs((S[j, k + 1] >= C[j, k] for j in job_ids for k in machines[:-1]), name="MachinePrecedenceConstr")

    # Add constraints for job ordering on each machine
    M = np.sum(processing_times) + np.max(release_dates)  # A large constant
    model.addConstrs((S[j, k] >= C[i, k] - M * (1 - x[i, j, k]) for i in job_ids for j in job_ids for k in machines if i != j), name="JobOrderConstr1")
    model.addConstrs((S[i, k] >= C[j, k] - M * x[i, j, k] for i in job_ids for j in job_ids for k in machines if i != j), name="JobOrderConstr2")

    # Add mutual exclusion constraints for job ordering
    model.addConstrs((x[i, j, k] + x[j, i, k] == 1 for i in job_ids for j in job_ids for k in machines if i != j), name="MutualExclusion")

    # Add constraints to calculate tardiness
    model.addConstrs((T[j] >= C[j, num_machines] - due_dates[j - 1] for j in job_ids), name="TardinessConstr")

    # Add precedence constraints to prioritize earlier jobs
    model.addConstrs(
        (S[i, 1] <= S[j, 1] for i, j in zip(job_ids[:-1], job_ids[1:]) if release_dates[i - 1] <= release_dates[j - 1]),
        name="PrecedenceByReleaseDate"
    )

    # Set the time limit for optimization
    model.Params.TimeLimit = max_runtime

    # Optimize the model
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        machine_schedules = {}
        for k in machines:
            schedule = []
            for j in job_ids:
                start_time = S[j, k].X
                completion_time = C[j, k].X
                schedule.append((j, start_time, completion_time))
            schedule.sort(key=lambda x: x[1])  # Sort by start time
            machine_schedules[k] = schedule

        return machine_schedules
    else:
        print("No feasible solution found.")
        return None

# Example usage
if __name__ == "__main__":
    # Input data
    data = readInput('data/job_data3.xlsx')

    # Run the algorithm
    MAX_RUNTIME = 10
    machine_schedules = runAlgorithm(data, MAX_RUNTIME)

    # Print the schedules
    if machine_schedules:
        for machine, schedule in machine_schedules.items():
            print(f"Schedule for Machine {machine}:")
            for job_id, start, complete in schedule:
                print(f"  Job {job_id}: Start at {start:.2f}, Complete at {complete:.2f}")
