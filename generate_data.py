import pandas as pd
import random

# Configuration variables
output_file = "data/job_data6.xlsx"
number_of_jobs = 1000
releases_date_range = 1000
min_due_date_range = 200
max_due_date_range = 1000
service_times_max = 5
weight_max = 10
num_machines = 4 # Number of random integers (service times) per job

# Generate sample data
data = []
for i in range(number_of_jobs):
    job_id = i + 1
    release_date = random.randint(0, releases_date_range)  # Integer release date starting at 0
    due_date = release_date + random.randint(min_due_date_range, max_due_date_range)  # Integer due date after release_date
    weight = round(random.randint(1, weight_max), 1)  # Random weight between 1.0 and weight_max
    
    # Generate random service times based on the number of machines
    service_times = {f"st_{j+1}": random.randint(1, service_times_max) for j in range(num_machines)}
    
    job_data = {
        "job_id": job_id,
        "release_date": release_date,
        "due_date": due_date,
        "weight": weight,
    }
    job_data.update(service_times)  # Add service times to the job data
    data.append(job_data)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel(output_file, index=False)

print(f"Data has been written to {output_file}")