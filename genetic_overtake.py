import numpy as np
import pandas as pd
import time
from genetic2 import runAlgorithmGen

def readInput(excel_file_path):
    '''
    This function reads data from an excel file and returns it.

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

def runAlgorithmGenO(data, npop = 10, gens = 100):
    '''
    An a genetic programming algorithm to solve the permutation flowshop scheduling
    problem with release dates and total weighted tardiness as a objective function.

    Input:
    - data -> a pandas dataframe containing the following column (in order):
        - job_id
        - release_date
        - due_date
        - weight
        - st_1 <-> st_m (with m machines)

    Output:
    - best_scores -> the best score of each of the generations
    - schedule -> the best schedule that the algorithm found
    - score -> the score of the best schedule
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

    # Start the timer
    start_time = time.time()

    # Generate random schedules
    schedules = generateRandomSchedules(num_jobs, num_machines, npop)

    # schedules = np.zeros((npop, num_machines, num_jobs))
    # init_schedules = runAlgorithmGen(data, npop, 1000)
    
    # for i in range(npop):
    #     schedules[i] = np.tile(init_schedules[i], (num_machines, 1))

    # Initialize an array for new schedules
    new_schedules = np.zeros_like(schedules)

    # Initialize array to store the best score of each generation
    best_scores = np.zeros(gens)

    # Loop over the generations
    for gen in range(gens):

        # Calulate the score of each schedule
        scores = np.array([calculateScore(schedule, machines, release_dates, due_dates, weights, processing_times)
                            for schedule in schedules])

        # Get the probability that each schedule is chosen
        probs = getProbabilitites(scores)

        # Stop if one of the schedules has 0 delay
        if np.min(scores) == 0:
            return 0, schedules[np.argmin(scores)]
        
        # Generate new schedules
        for i in range(npop - 1):
            
            # Get the two parent schedules
            s1, s2 = np.random.choice(npop, 2, replace=False, p = probs)

            # Perform the crossover between the two parent schedules
            schedule = crossoverSchedules(schedules[s1], schedules[s2])
            #schedule = schedules[s1].copy()

            # Perform a mutation and add to the new schedules
            new_schedules[i] = mutatateSchedule(schedule)

        # Add the best schedule of the previous generation (Elitist strategy)
        new_schedules[npop - 1] = schedules[np.argmin(scores)]

        # Assign the new schedules to actual schedules
        schedules = new_schedules.copy()

        # Get the best score of the previous generation
        best_scores[gen] = np.min(scores)

    # Get scores of the last generation
    scores = np.array([calculateScore(schedule, machines, release_dates, due_dates, weights, processing_times)
                            for schedule in schedules])
    
    # Get the lowest score
    min_score = np.min(scores)

    # Get the best schedule
    best_schedule = schedules[np.argmin(scores)]
    best_schedule_df = scheduleToDf(best_schedule, machines, release_dates, processing_times)

    # Get the exact time
    exact_time = time.time() - start_time

    return best_schedule_df, min_score, exact_time, best_scores, best_schedule 

def scheduleToDf(schedule, machines, release_dates, processing_times):

    # Initialize completion times array
    num_jobs = schedule.shape[1]
    completion_times = np.zeros_like(processing_times, dtype=int)
    current_time = 0

    # Get the completion times of all jobs on the first machine
    for job in schedule[0]:
        job = int(job)

        if release_dates[job - 1] > current_time:
            current_time = release_dates[job - 1]

        completion_times[job - 1, 0] = current_time + processing_times[job - 1, 0]
        current_time = completion_times[job - 1, 0]

    # Get the remaining compeletion times
    for machine in machines[1:]:
        machine -= 1

        for j in range(len(schedule)):
            job = int(schedule[machine, j])

            if j == 0:
                C = completion_times[job - 1, machine - 1]
            else:
                C = np.max([completion_times[int(schedule[machine, j - 1] - 1), machine], completion_times[job - 1, machine - 1]])

            completion_times[job - 1, machine] = C + processing_times[job - 1, machine]

    # Append job schedule to DataFrame
    start_times = completion_times - processing_times
    data  = pd.DataFrame({'Job ID': [i for i in range(1, num_jobs + 1)]})

    for m in machines:
        data[f'Start time machine {m}'] = start_times[:, m - 1]
        data[f'Completion time machine {m}'] = completion_times[:, m - 1]
    
    data = data.sort_values(by='Job ID')
    return data

def calculateScore(schedule, machines, release_dates, due_dates, weights, processing_times):
    '''
    A function that calculates the total weighted tardiness of the given schedule.

    Input:
    - schedule -> an array containing the order in which the jobs are processed
    - machines -> an array containing an index for all of the machines, i.e. [0, 1, ..., m] for m machines
    - release_dates -> an array containing the release dates of each job
    - due_dates -> an array containig the due dates of each job
    - weights -> an array containing the weight of each job
    - processing_times -> an n x m matrix containing the the processing times of each job on each machine

    Output:
    - score -> the total weighted tardiness of the schedule
    '''
    
    # Define variables
    completion_times = np.zeros_like(processing_times)
    current_time = 0

    # Get the completion times of all jobs on the first machine
    for job in schedule[0]:
        job = int(job)

        if release_dates[job - 1] > current_time:
            current_time = release_dates[job - 1]

        completion_times[job - 1, 0] = current_time + processing_times[job - 1, 0]
        current_time = completion_times[job - 1, 0]

    # Get the remaining compeletion times
    for machine in machines[1:]:
        machine -= 1

        for j in range(len(schedule[machine])):
            job = int(schedule[machine ,j])

            if j == 0:
                C = completion_times[job - 1, machine - 1]
            else:
                C = np.max([completion_times[int(schedule[machine, j - 1] - 1), machine], completion_times[job - 1, machine - 1]])

            completion_times[job - 1, machine] = C + processing_times[job - 1, machine]

    # Calculate the weighted tardiness of each job
    tardiness = np.clip((completion_times[:, machines[-2]] - due_dates) * weights, a_min = 0, a_max = None)

    # Calculate the total weighted tardiness
    score = np.sum(tardiness)

    return score

def generateRandomSchedules(num_jobs, num_machines, num_schedules):
    '''
    A function that creates num_schedules random schedules, each containing num_jobs jobs.

    input:
    - num_jobs -> the number of jobs in each schedule
    - num_schedules -> the number of schedules that need to be generated
    '''

    # Initialize random schedules
    schedules = np.zeros((num_schedules, num_machines, num_jobs))

    # Generate the random schedules'
    for i in range(num_schedules):
        schedule = np.random.permutation(num_jobs) + 1
        schedule = np.tile(schedule, (num_machines, 1))

        schedules[i] = schedule

    return schedules

def crossoverSchedules(schedule1, schedule2):
    '''
    Combine two different schedules based on a two-point crossover technique to create a new schedule.

    input:
    - schedule1 -> an array containing the order in which jobs are processed
    - schedule2 -> an array containing the order in which jobs are processed
    '''

    # new_schedule = schedule1.copy()

    # m, n = schedule1.shape
    # points = np.random.choice(n + 1, 2)
    # machine = np.random.choice(m)

    # point1 = np.min(points)
    # point2 = np.max(points)

    # for m in range(machine, m):
    #     missing_jobs = set(schedule1[m, point1:point2])

    #     new = np.array([x for x in schedule2[m] if x in missing_jobs])

    #     new_schedule[m] = np.concatenate([schedule1[m, :point1], new, schedule1[m, point2:]])

    # Sample random point
    m = schedule1.shape[0]
    index = np.random.choice(m)

    # Combine schedule
    new_schedule = schedule1.copy()
    new_schedule[index:] = schedule2[index:]

    return new_schedule

def mutatateSchedule(schedule):
    '''
    This function performs one shift mutation to a schedule.

    input:
    - schedule -> an array containing the order in which jobs are processed

    output:
    - schedule -> an array containing the mutated order in which jobs are processed
    '''
    
    m, n = schedule.shape
    job = np.random.choice(n) + 1
    machine = np.random.choice(m)

    indices = np.where(schedule == job)[1]
    new_index = np.random.choice(n)
    
    new_schedule = schedule.copy()

    for i in range(machine, m):
        part = np.delete(new_schedule[i], indices[i])
        part = np.insert(part, new_index, job)

        new_schedule[i] = part

    return new_schedule

def getProbabilitites(scores):
    '''
    This function calculates the probabilities of each of the schedule
    being chosen based on their score. It uses the squared distance to
    the worst solution and normalizes that.

    input:
    - scores -> an array that contains the score of each of the schedules

    output:
    - probs -> the probabilities that each of the schedules is chosen
    '''

    # Get the worst score
    worst_score = np.max(scores)

    # Get the total squared distance to the worst score
    denominator = np.sum((worst_score - scores)**2)

    # Calculate the probabilities
    probs = (worst_score - scores)**2 / denominator

    return probs

if __name__ == "__main__":
    data = readInput('data/job_data2.xlsx')

    start = time.time()
    df, min_score, exact_time, best_scores, best_schedule = runAlgorithmGenO(data, npop = 10, gens = 1000)
    end = time.time()

    print('Schedule:\n', best_schedule)
    print('Score:', min_score)
    print('Runtime:', end - start)

    print('List of scores:', best_scores)
