import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from datetime import datetime
import autogen
from typing import Union, Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import requests
from autogen.cache import Cache

app = Flask(__name__)

# Model parameters
input_length = 30
hidden_size = 64
dropout = 0.3
num_layers = 2

# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = nn.Embedding(input_length, hidden_size)
        self.transformer = nn.Transformer(hidden_size, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x

# Load the trained model
model_folder_path = '/home/ubuntu/efs-w210-capstone-ebs/04B.Local_Model_Files/20240713_Non_Overlapping_Consl_180_FINAL'
trained_model_file = f"{model_folder_path}/20240712_Transformers_Non_Overlapping_run_143_of_180.pt"
model = TransformerModel(input_size=55, hidden_size=64, num_layers=2, output_size=1, dropout=0.3)
model.load_state_dict(torch.load(trained_model_file))
model.eval()

# Autogen setup
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": "sk-proj-iQtcgUJOOf4n53Bs6uyqT3BlbkFJnEIqUeEwXjbjVMcDVqiz"}],
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "group",
        "use_docker": False,
    },
    max_consecutive_auto_reply=10,
    system_message="""Reply TERMINATE if the task been solved at full satisfaction. 
    Otherwise, reply CONTINUE or the reason why the task is not solved yet. """,
)

log_data_retriever = autogen.AssistantAgent(
    name="log_data_retriever",
    system_message="For all tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

log_data_writer = autogen.AssistantAgent(
    name="log_data_writer",
    system_message="""
    You are a creative writer and your job is to take the data extracted by log_data_retriever and 
    summarize the content in a email with subject header and body. Please keep the email concise and focus on the analysis of the confusion matrix.
    Specifically since this is a log anomaly detection task, please try to analyze the recall and precision of Class 1 (ie Alert)
    """,
    llm_config=llm_config,
)

def send_log_data_and_get_model_results(start_time: str, num_tests: int) -> Union[dict, str]:
    # Data paths
    data_dir = '/home/ubuntu/efs-w210-capstone-ebs/04A.Local_Data_Files'
    save_dir = '/home/ubuntu/efs-w210-capstone-ebs/04C.Local_Inference_Eval_Files'
    file_prefix = "06.20240714_062624_non_overlap_full_test"

    # Load test data
    X_test = np.load(f"{data_dir}/{file_prefix}_X_test.npy")
    y_test = np.load(f"{data_dir}/{file_prefix}_y_test.npy")

    # Load original parquet file with sequence mapping
    input_file = f"{data_dir}/03.20240715_143154_orig_input_w_seq_info_FINAL.parquet"
    original_df = pd.read_parquet(input_file)
    original_df['Seq_Num'] = original_df['Seq_Num'].astype(int)

    # Filter test dataset based on start time
    filtered_df = original_df[(original_df['Train_Test'] == 'Test') & (original_df['time_start'] == start_time)]
    if filtered_df.empty:
        # If the specific start time is not found, find other available times on the same date
        start_date = pd.to_datetime(start_time).date()
        same_date_df = original_df[(original_df['Train_Test'] == 'Test') & (pd.to_datetime(original_df['time_start']).dt.date == start_date)]
        
        if not same_date_df.empty:
            available_times = same_date_df['time_start'].unique().tolist()
            return f"Error: start_time {start_time} not found in the dataset. Available times on {start_date} are: {available_times}"
        
        # If no times are available on the same date, find the nearest available time
        all_times_df = original_df[original_df['Train_Test'] == 'Test']
        all_times_df['time_start'] = pd.to_datetime(all_times_df['time_start'])
        nearest_time = all_times_df.iloc[(all_times_df['time_start'] - pd.to_datetime(start_time)).abs().argsort()[:1]]['time_start'].values[0]
        return f"Error: start_time {start_time} not found in the dataset. The nearest available time is: {nearest_time}"

    start_seq_num = filtered_df['Seq_Num'].values[0]
    
    num_records_per_test = 33
    max_end_seq_num = original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].max()
    expected_end_seq_num = start_seq_num + num_tests * num_records_per_test - 1

    if expected_end_seq_num > max_end_seq_num:
        max_allowed_runs = (max_end_seq_num - start_seq_num + 1) // num_records_per_test
        num_tests = max_allowed_runs
        expected_end_seq_num = start_seq_num + num_tests * num_records_per_test - 1

    # Calculate correct start and end sequence numbers for X_test and y_test
    start_index_x_test = (start_seq_num - original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].min()) // num_records_per_test
    end_index_x_test = min(start_index_x_test + num_tests, len(X_test))

    # Prepare data for specified number of test runs
    X_test_limited = X_test[start_index_x_test:end_index_x_test]
    y_test_limited = y_test[start_index_x_test:end_index_x_test]

    # Prepare data for request
    data = {'X_test': X_test_limited.tolist()}

    # Send request to server
    response = requests.post('http://0.0.0.0:5000/predict', json=data)
    if response.status_code != 200:
        return f"Failed to get predictions: Status Code {response.status_code}"

    response_data = response.json()
    predictions = np.array(response_data['predictions']).flatten()

    # Create DataFrame for tracking and merging with additional information
    tracking_data = []
    for i in range(num_tests):
        total_seq_start = start_seq_num + i * num_records_per_test
        total_seq_end = total_seq_start + num_records_per_test - 1
        source_seq_start = total_seq_start
        source_seq_end = source_seq_start + 30 - 1
        gap_seq_start = source_seq_end + 1
        gap_seq_end = gap_seq_start + 1
        prediction_seq_start = total_seq_end - 1
        prediction_seq_end = total_seq_end
        
        tracking_data.append([
            total_seq_start, total_seq_end, source_seq_start, source_seq_end,
            gap_seq_start, gap_seq_end, prediction_seq_start, prediction_seq_end,
            predictions[i], y_test_limited[i],
            str(original_df[original_df['Seq_Num'] == source_seq_start]['time_start'].values[0])
        ])

    tracking_df = pd.DataFrame(tracking_data, columns=[
        "Total_Seq_Start", "Total_Seq_End", "Source_Seq_Start", "Source_Seq_End",
        "Gap_Seq_Start", "Gap_Seq_End", "Prediction_Seq_Start", "Prediction_Seq_End",
        "Predicted", "Actual", "time_start"
    ])

    # Save the consolidated DataFrame
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    final_file = f"{save_dir}/03B.{timestamp}_agent1_non_overlap_model2_consl.csv"
    tracking_df.to_csv(final_file, index=False)

    # Calculate metrics
    conf_matrix = confusion_matrix(y_test_limited, predictions, labels=[0, 1])
    precision = precision_score(y_test_limited, predictions, zero_division=0)
    recall = recall_score(y_test_limited, predictions, zero_division=0)
    accuracy = accuracy_score(y_test_limited, predictions)
    f1 = f1_score(y_test_limited, predictions, zero_division=0)

    if conf_matrix.size == 4:
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
        tn = conf_matrix[0, 0] if conf_matrix.shape[0] > 0 else 0
        fp = conf_matrix[0, 1] if conf_matrix.shape[1] > 1 else 0
        fn = conf_matrix[1, 0] if conf_matrix.shape[0] > 1 else 0
        tp = conf_matrix[1, 1] if conf_matrix.shape[1] > 1 else 0

    # Prepare results
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
        "output_file": final_file
    }

    return results

# Register function for execution
user_proxy.register_for_execution()(send_log_data_and_get_model_results)
log_data_retriever.register_for_llm(description="Run inference on test data and evaluate model performance")(send_log_data_and_get_model_results)

# Setup group chat
groupchat = autogen.GroupChat(agents=[user_proxy, log_data_retriever, log_data_writer], messages=[], max_round=10)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        start_time = request.form['start_time']
        num_tests = int(request.form['num_tests'])
        
        task = f"""
        Run the send_log_data_and_get_model_results function with the following parameters:
        start_time = '{start_time}'
        num_tests = {num_tests}
        """
        
        with Cache.disk() as cache:
            chat_result = user_proxy.initiate_chat(
                manager, message=task, summary_method="reflection_with_llm", cache=cache, llm_config=llm_config
            )
        
        # Extract results from the chat response
        results = None
        for message in chat_result.chat_history:
            if isinstance(message, dict) and 'content' in message:
                try:
                    content = message['content']
                    if isinstance(content, str) and content.startswith('{') and content.endswith('}'):
                        results = eval(content)
                        if isinstance(results, dict) and 'metrics' in results:
                            break
                except:
                    continue
        
        if results is None:
            return "Failed to get results from the model"
        
        metrics = results['metrics']
        confusion_matrix = results['confusion_matrix']
        output_file = results['output_file']
        
        return render_template('20240718_claude_results_v1.00.html', metrics=metrics, confusion_matrix=confusion_matrix, output_file=output_file)
    
    return render_template('20240718_claude_index_v1.00.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_test = np.array(data['X_test'])
        X_test = X_test[:, :, 1:]  # Drop the first column
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_outputs_sigmoid = torch.sigmoid(test_outputs)
            test_preds = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).tolist()
        
        return jsonify(predictions=test_preds)
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)