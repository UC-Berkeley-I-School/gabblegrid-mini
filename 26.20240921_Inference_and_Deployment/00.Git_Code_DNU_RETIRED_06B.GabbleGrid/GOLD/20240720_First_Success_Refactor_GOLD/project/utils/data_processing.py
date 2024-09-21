import numpy as np
import pandas as pd
import torch

def load_test_data(data_dir, file_prefix):
    X_test = np.load(f"{data_dir}/{file_prefix}_X_test.npy")
    y_test = np.load(f"{data_dir}/{file_prefix}_y_test.npy")
    return X_test, y_test

def filter_data_by_start_time(original_df, start_time):
    original_df['Seq_Num'] = original_df['Seq_Num'].astype(int)
    filtered_df = original_df[(original_df['Train_Test'] == 'Test') & (original_df['time_start'] == start_time)]
    return filtered_df

def prepare_data_for_model(X_test, start_seq_num, num_records_per_test, num_tests):
    start_index_x_test = (start_seq_num - original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].min()) // num_records_per_test
    end_index_x_test = min(start_index_x_test + num_tests, len(X_test))
    X_test_limited = X_test[start_index_x_test:end_index_x_test]
    X_test_limited = X_test_limited[:, :, 1:]  # Drop the first column
    X_test_tensor = torch.tensor(X_test_limited, dtype=torch.float32)
    return X_test_tensor
