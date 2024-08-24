# File: agents/historical_weather/data_preparation.py

import torch
import pandas as pd
import re

#################### Change needed to fix index out of bounds error #######################
def prepare_data_for_model(X_test, start_seq_num, num_records_per_test, num_tests, original_df, max_events):
    start_index_x_test = (start_seq_num - original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].min()) // num_records_per_test
    end_index_x_test = min(start_index_x_test + num_tests, len(X_test))
    
    # Adjust num_tests if the end index is out of bounds
    if end_index_x_test > len(X_test):
        num_tests = len(X_test) - start_index_x_test
        end_index_x_test = start_index_x_test + num_tests
    
    X_test_limited = X_test[start_index_x_test:end_index_x_test]
    X_test_limited = X_test_limited[:, :, 1:max_events + 16]
    X_test_tensor = torch.tensor(X_test_limited, dtype=torch.float32)
    
    return X_test_tensor, start_index_x_test, end_index_x_test
############################################################################################


def consolidate_events_to_text(df, start_col, end_col, col_name, master_tracking_df, eventid_to_template):
    consolidated = []
    for i, row in df.iterrows():
        seen_events = set()
        events = []
        for seq_num in range(row[start_col], row[end_col] + 1):
            if col_name in master_tracking_df.columns:
                event_list = master_tracking_df[master_tracking_df['Seq_Num'] == seq_num][col_name].astype(str).tolist()
                for event in event_list:
                    for e in event.split(', '):
                        if e not in seen_events:
                            seen_events.add(e)
                            if int(e) in eventid_to_template:
                                cleaned_text = clean_text(eventid_to_template[int(e)])
                                events.append(f'"{cleaned_text}"')
                            else:
                                events.append('"Unknown"')
        consolidated.append(', '.join(events))
    return consolidated

def clean_text(text):
    text = re.sub(r'<\*?>', '', text)
    text = text.replace('<', '').replace('>', '')
    return text

def get_latest_experiment_id(parquet_file):
    """
    Retrieves the latest experiment ID from the provided Parquet file.

    :param parquet_file: Path to the Parquet file containing experiment data.
    :return: The latest experiment ID.
    """
    df = pd.read_parquet(parquet_file)
    if not df.empty:
        latest_experiment = df['Experiment'].max()  # Assuming 'Experiment' column exists
        return latest_experiment
    else:
        raise ValueError("The experiment file is empty.")