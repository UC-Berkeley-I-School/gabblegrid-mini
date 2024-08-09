import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from datetime import datetime
from typing import Union
from .data_processing import load_test_data, filter_data_by_start_time, prepare_data_for_model
from ..beta_model.load_model import load_model  # Update import path
import logging

def send_log_data_and_get_model_results(start_time: str, num_tests: int, input_length: int, gap: int, prediction_period: int, max_events: int, csv_path: str) -> Union[dict, str]:
    try:
        model, model_params = load_model(input_length, gap, prediction_period, max_events, csv_path)
        
        data_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/02.Local_Data_Files'
        save_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/03.Local_Inference_Eval_Files'
        file_prefix = "06.20240714_062624_non_overlap_full_test"

        X_test, y_test = load_test_data(data_dir, file_prefix)

        input_file = f"{data_dir}/03.20240715_143154_orig_input_w_seq_info_FINAL.parquet"
        original_df = pd.read_parquet(input_file)
        original_df['Seq_Num'] = original_df['Seq_Num'].astype(int)

        filtered_df = filter_data_by_start_time(original_df, start_time)
        if filtered_df.empty:
            start_date = pd.to_datetime(start_time).date()
            same_date_df = original_df[(original_df['Train_Test'] == 'Test') & (pd.to_datetime(original_df['time_start']).dt.date == start_date)]
            
            if not same_date_df.empty:
                available_times = same_date_df['time_start'].unique().tolist()
                return f"Error: start_time {start_time} not found in the dataset. Available times on {start_date} are: {available_times}"
            
            all_times_df = original_df[original_df['Train_Test'] == 'Test']
            all_times_df['time_start'] = pd.to_datetime(all_times_df['time_start'])
            nearest_time = all_times_df.iloc[(all_times_df['time_start'] - pd.to_datetime(start_time)).abs().argsort()[:1]]['time_start'].values[0]
            return f"Error: start_time {start_time} not found in the dataset. The nearest available time is: {nearest_time}"

        start_seq_num = int(filtered_df['Seq_Num'].values[0])  # Ensure start_seq_num is an integer
        
        num_records_per_test = input_length + gap + prediction_period
        max_end_seq_num = int(original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].max())  # Ensure max_end_seq_num is an integer
        expected_end_seq_num = start_seq_num + num_tests * num_records_per_test - 1

        if expected_end_seq_num > max_end_seq_num:
            max_allowed_runs = (max_end_seq_num - start_seq_num + 1) // num_records_per_test
            num_tests = max_allowed_runs
            expected_end_seq_num = start_seq_num + num_tests * num_records_per_test - 1

        # Use the prepare_data_for_model function to process the test data
        X_test_tensor, start_index_x_test, end_index_x_test = prepare_data_for_model(X_test, start_seq_num, num_records_per_test, num_tests, original_df, max_events)

        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_outputs_sigmoid = torch.sigmoid(test_outputs)
            predictions = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).flatten()

        tracking_data = []
        for i in range(num_tests):
            total_seq_start = start_seq_num + i * num_records_per_test
            total_seq_end = total_seq_start + num_records_per_test - 1
            source_seq_start = total_seq_start
            source_seq_end = source_seq_start + input_length - 1
            gap_seq_start = source_seq_end + 1
            gap_seq_end = gap_seq_start + gap - 1
            prediction_seq_start = total_seq_end - prediction_period + 1
            prediction_seq_end = total_seq_end
            
            tracking_data.append([
                total_seq_start, total_seq_end, source_seq_start, source_seq_end,
                gap_seq_start, gap_seq_end, prediction_seq_start, prediction_seq_end,
                predictions[i], y_test[start_index_x_test + i],
                str(original_df[original_df['Seq_Num'] == source_seq_start]['time_start'].values[0])
            ])

        tracking_df = pd.DataFrame(tracking_data, columns=[
            "Total_Seq_Start", "Total_Seq_End", "Source_Seq_Start", "Source_Seq_End",
            "Gap_Seq_Start", "Gap_Seq_End", "Prediction_Seq_Start", "Prediction_Seq_End",
            "Predicted", "Actual", "time_start"
        ])

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        final_file = f"{save_dir}/03B.{timestamp}_agent1_non_overlap_model2_consl.csv"
        tracking_df.to_csv(final_file, index=False)

        conf_matrix = confusion_matrix(y_test[start_index_x_test:end_index_x_test], predictions, labels=[0, 1])
        precision = precision_score(y_test[start_index_x_test:end_index_x_test], predictions, zero_division=0)
        recall = recall_score(y_test[start_index_x_test:end_index_x_test], predictions, zero_division=0)
        accuracy = accuracy_score(y_test[start_index_x_test:end_index_x_test], predictions)
        f1 = f1_score(y_test[start_index_x_test:end_index_x_test], predictions, zero_division=0)

        if conf_matrix.size == 4:
            tn, fp, fn, tp = conf_matrix.ravel()
        else:
            tn = conf_matrix[0, 0] if conf_matrix.shape[0] > 0 else 0
            fp = conf_matrix[0, 1] if conf_matrix.shape[1] > 1 else 0
            fn = conf_matrix[1, 0] if conf_matrix.shape[0] > 1 else 0
            tp = conf_matrix[1, 1] if conf_matrix.shape[1] > 1 else 0

        results = {
            "metrics": {
                "Accuracy": accuracy,
                "Precision (Class 1)": precision,
                "Recall (Class 1)": recall,
                "F1 Score": f1,
                "True Positives": int(tp),
                "False Positives": int(fp),
                "True Negatives": int(tn),
                "False Negatives": int(fn)
            },
            "confusion_matrix": conf_matrix.tolist(),
            "output_file": final_file,
            "model_params": model_params
        }

        # Log the model parameters and the selected model
        logging.info(f"Model parameters: {model_params}")
        
        return results

    except Exception as e:
        logging.error(f"Error in send_log_data_and_get_model_results: {str(e)}")
        return {"error": str(e)}