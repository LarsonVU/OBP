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
    C = model.addVars(job_ids, machines, lb=0, vtype=GRB.CONTINUOUS, name = "CompletionTime")

    # Initialize variables that indicate the tardiness of job j
    T = model.addVars(job_ids, lb=0, vtype=GRB.CONTINUOUS, name = "Tardiness")

    # Initialize variable that indicate if job i is before job j in the order
    x = model.addVars(job_ids, job_ids, vtype=GRB.BINARY, name = "Order")

    # Set the object to minimize the total weighted tardiness
    model.setObjective(sum(weights[j - 1] * T[j] for j in job_ids), GRB.MINIMIZE)

    # Add constraint that ensures that job j starts after its release date r_j
    model.addConstrs((C[j, 1] >= release_dates[j - 1] + processing_times[j - 1, 0] for j in job_ids), name = "ReleaseDateConstr")

    # Add constraint that ensures that job j on machine k (>= 2) start after it has been processed on machine k - 1
    model.addConstrs((C[j, k] >= C[j, k - 1] + processing_times[j - 1, k - 1]
                       for j in job_ids for k in machines[1:]), name = 'MachinePrecedenceConstr')

    # Calculate an upperbound on the 
    M = np.sum(processing_times) + np.max(release_dates)

    # Add constraint that ensures that if job j happens after job i its completion time is larger
    model.addConstrs((C[j, k] >= C[i, k] + processing_times[j - 1, k - 1] * x[i, j] - M * (1 - x[i, j]) 
                      for i in job_ids for j in job_ids for k in machines if i != j), name = "JobOrderConstr")

    # Add constraint that calculates the tardiness
    model.addConstrs((T[j] >= C[j, num_machines] - due_dates[j - 1] for j in job_ids), name = "TardinessConstr")

    # Add constraint that ensures that job j cannot happens before itself
    model.addConstrs((x[j, j] == 0 for j in job_ids), name = "NoSelfPrecedenceConstr")

    # Add constraint that ensures that either job i happens before job j or vice versa if i!=j
    model.addConstrs((x[i, j] + x[j, i] == 1 for i in job_ids for j in job_ids if i != j), name = "MutualExclusion")

    # Set timelimit parameter
    model.Params.TimeLimit = max_runtime

    # Find schedule
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        columns = ['Job ID']
        for i in range(len(machines)):
            columns = columns + [f'Start time machine {i+1}', f'Completion time machine {i+1}' ]
        schedule = pd.DataFrame(columns=columns)

        for j in job_ids:
            job_schedule = [j]

            for k in machines:
                job_schedule = job_schedule + [C[j, k].X - processing_times[j - 1, k-1] , C[j, k].X]

            schedule.loc[len(schedule)] = job_schedule

        score = model.objVal
        return schedule, score, model.Runtime
    else:
        print("No feasible solution found.")
        return None, None, None

# Example usage
if __name__ == "__main__":
    data = readInput('data/job_data3.xlsx')
    MAX_RUNTIME = 10
    schedule, score, runtime = runAlgorithm(data, MAX_RUNTIME)
    print("Schedule:\n", schedule)
    print("Score:", score)
    print("Runtime:", runtime)