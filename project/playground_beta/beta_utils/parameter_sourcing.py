import pandas as pd

def get_time_range(input_file):
    # Load the input file
    original_df = pd.read_parquet(input_file)
    # Filter for 'Test' records
    test_df = original_df[original_df['Train_Test'] == 'Test'].copy()  # Add .copy() to avoid SettingWithCopyWarning
    # Convert 'time_start' to datetime
    test_df['time_start'] = pd.to_datetime(test_df['time_start'])
    # Get the minimum and maximum time
    min_time = test_df['time_start'].min()
    max_time = test_df['time_start'].max()
    return min_time, max_time
