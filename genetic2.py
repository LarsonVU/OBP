import numpy as np
import pandas as pd
import time

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

def runAlgorithmGen(data, npop = 10, gens = 100):
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
    #start the timer
    start_time = time.time()

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

    # Generate random schedules
    schedules = generateRandomSchedules(num_jobs, npop)

    # Initialize an array for new schedules
    new_schedules = np.zeros((npop, num_jobs), dtype = int)

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
            break #return 0, schedules[np.argmin(scores)]
        
        # Generate new schedules
        for i in range(npop - 1):
            
            # Get the two parent schedules
            s1, s2 = np.random.choice(npop, 2, replace=False, p = probs)

            # Perform the crossover between the two parent schedules
            schedule = crossoverSchedules(schedules[s1], schedules[s2])

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

    return schedules
    return best_schedule_df, min_score, exact_time, best_scores, best_schedule 

def scheduleToDf(schedule, machines, release_dates, processing_times):

    # Initialize completion times array
    num_jobs = len(schedule)
    completion_times = np.zeros_like(processing_times, dtype=int)
    current_time = 0

    # Get the completion times of all jobs on the first machine
    for job in schedule:
        if release_dates[job - 1] > current_time:
            current_time = release_dates[job - 1]

        completion_times[job - 1, 0] = current_time + processing_times[job - 1, 0]
        current_time = completion_times[job - 1, 0]

    # Get the remaining compeletion times
    for machine in machines[1:]:
        machine -= 1

        for j in range(len(schedule)):
            job = schedule[j]

            if j == 0:
                C = completion_times[job - 1, machine - 1]
            else:
                C = np.max([completion_times[schedule[j - 1] - 1, machine], completion_times[job - 1, machine - 1]])

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
    for job in schedule:
        if release_dates[job - 1] > current_time:
            current_time = release_dates[job - 1]

        completion_times[job - 1, 0] = current_time + processing_times[job - 1, 0]
        current_time = completion_times[job - 1, 0]

    # Get the remaining compeletion times
    for machine in machines[1:]:
        machine -= 1

        for j in range(len(schedule)):
            job = schedule[j]

            if j == 0:
                C = completion_times[job - 1, machine - 1]
            else:
                C = np.max([completion_times[schedule[j - 1] - 1, machine], completion_times[job - 1, machine - 1]])

            completion_times[job - 1, machine] = C + processing_times[job - 1, machine]

    # Calculate the weighted tardiness of each job
    tardiness = np.clip((completion_times[:, machines[-2]] - due_dates) * weights, a_min = 0, a_max = None)

    # Calculate the total weighted tardiness
    score = np.sum(tardiness)

    return score

def generateRandomSchedules(num_jobs, num_schedules):
    '''
    A function that creates num_schedules random schedules, each containing num_jobs jobs.

    input:
    - num_jobs -> the number of jobs in each schedule
    - num_schedules -> the number of schedules that need to be generated
    '''

    # Generate the random schedules
    schedules = np.array([np.random.permutation(num_jobs) + 1 for _ in range(num_schedules)])

    return schedules

def crossoverSchedules(schedule1, schedule2):
    '''
    Combine two different schedules based on a two-point crossover technique to create a new schedule.

    input:
    - schedule1 -> an array containing the order in which jobs are processed
    - schedule2 -> an array containing the order in which jobs are processed
    '''

    # Sample two random points in the schedule
    points = np.random.choice(np.arange(0, len(schedule1) + 1), 2)

    # Set point1 to the lowest point and point2 to the highest
    point1 = np.min(points)
    point2 = np.max(points)

    # Get the jobs between the two points
    missing_jobs = set(schedule1[point1:point2])

    # Get the order in which these jobs appear in schedule2
    new = np.array([x for x in schedule2 if x in missing_jobs])

    # Combine the part outside of the two points in schedule1 with the order
    # in which the missing jobs appear in schedule 2
    new_schedule = np.concatenate([schedule1[:point1], new, schedule1[point2:]])

    return new_schedule

def mutatateSchedule(schedule):
    '''
    This function performs one shift mutation to a schedule.

    input:
    - schedule -> an array containing the order in which jobs are processed

    output:
    - schedule -> an array containing the mutated order in which jobs are processed
    '''
    
    # Sample 2 random indices: index1 is the index of the job,
    # index2 is where this job will be placed
    index1, index2 = np.random.choice(np.arange(0, len(schedule)), 2, replace=False)

    # Get the job
    job = schedule[index1]

    # Remove the job from the schedule
    new_schedule = np.delete(schedule, index1)

    # Place it back at the new index
    new_schedule = np.insert(new_schedule, index2, job)

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
    data = readInput('data/job_data4.xlsx')

    start = time.time()
    df, min_score, exact_time, best_scores, best_schedule = runAlgorithmGen(data, npop = 10, gens = 100)
    end = time.time()

    print('Schedule:\n', best_schedule)
    print('Score:', min_score)
    print('Runtime:', end - start)

    print('List of scores:', best_scores)

# npop = 10
# gens = 100
# data = readInput('data/job_data6.xlsx')
# start = time.time()
# print(runAlgorithmGen(data, npop, gens))
# end = time.time()

# print(end-start)