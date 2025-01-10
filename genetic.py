import numpy as np
import pandas as pd
import time

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

def runAlgorithm(data, npop = 10, gens = 100):
    '''
    An a genetic programming algorithm to solve the permutation flowshop scheduling
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

    times = np.ones(1000)

    return np.mean(times)

def calculateScore(schedule, machines, release_dates, due_dates, weights, processing_times):
    schedule -= 1
    current_time = 0
    completion_times = np.zeros(processing_times.shape)

    for job in schedule:
        completion_times[job, 0] = current_time + processing_times[job, 0]

        if release_dates[job] > current_time:
            completion_times[job, 0] = release_dates[job] + processing_times[job, 0] 

        current_time = completion_times[job, 0]

    for machine in machines[1:]:
        machine -= 1

        for j in range(len(schedule)):
            job = schedule[j]

            if j == 0:
                C = completion_times[job, machine - 1]
            else:
                C = np.max([completion_times[schedule[j - 1], machine], completion_times[job, machine - 1]])

            completion_times[job, machine] = C + processing_times[job, machine]
    return np.sum((completion_times[:, machine] - due_dates) * weights)

def generateRandomSchedules(num_jobs, num_schedules):
    return np.array([np.random.permutation(num_jobs) + 1 for _ in range(num_schedules)])

def combineSchedules(schedule1, schedule2):
    point1 = np.random.randint(0, len(schedule1 + 1))
    point2 = np.random.randint(point1, len(schedule1 + 2))

    missing_jobs = set(schedule1[point1:point2])
    new = np.array([x for x in schedule2 if x in missing_jobs])

    return np.concatenate([schedule1[:point1], new, schedule1[point2:]])

# npop = 10
# gens = 100
# data = readInput('OBP/job_data3.xlsx')
# print(runAlgorithm(data, npop, gens))

schedule1 = [1,2,3,4,5,6,7,8]
schedule2 = [8,7,6,5,4,3,2,1]
point1 = 2
point2 = 5

# st = set(schedule1[point1:point2])
# print([x for x in schedule2 if x in st])
#print(combineSchedules(schedule1, schedule2))
res = np.zeros(10000)
for i in range(10000):
    point1 = np.random.randint(0, len(schedule1) + 1)
    point2 = np.random.randint(point1, len(schedule1) + 1)

    length =len(schedule1[point1:point2])
    res[i] = length

    if length == 0:
        print(point1, point2)

# res = np.zeros(10000)
# for i in range(10000):
#     points = np.random.choice(np.arange(0, 9), 2)
#     point1 = np.min(points)
#     point2 = np.max(points)

#     length =len(schedule1[point1:point2])
#     res[i] = length

import matplotlib.pyplot as plt
print(np.min(res))
plt.hist(res)
plt.show()