import pandas as pd

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