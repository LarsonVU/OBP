import pandas as pd
import random

# Generate sample data
output_file = "job_data2.xlsx"
number_of_jobs = 20
releases_date_range = 20
due_date_range = 20
service_times_max = 10
weight_max = 10

data = []
for i in range(number_of_jobs):
    job_number = i + 1
    release_date = random.randint(0, releases_date_range)  # Integer release date starting at 0
    due_date = release_date + random.randint(1, due_date_range)  # Integer due date after release_date
    weight = round(random.randint(1, 10), 1)  # Random weight between 1.0 and 5.0
    st_1 = random.randint(1, service_times_max)  # Random integer for st_1
    st_2 = random.randint(1, service_times_max)  # Random integer for st_2
    st_3 = random.randint(1, service_times_max)  # Random integer for st_3
    
    data.append({
        "job_number": job_number,
        "release_date": release_date,
        "due_date": due_date,
        "weight": weight,
        "st_1": st_1,
        "st_2": st_2,
        "st_3": st_3
    })

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel(output_file, index=False)

print(f"Data has been written to {output_file}")