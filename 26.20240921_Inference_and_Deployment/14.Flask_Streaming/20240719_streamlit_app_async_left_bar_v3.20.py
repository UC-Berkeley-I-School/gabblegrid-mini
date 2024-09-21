import streamlit as st
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from datetime import datetime
import autogen
from typing import Union, Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import requests
from autogen.cache import Cache
import asyncio
import json
import time
import networkx as nx
import matplotlib.pyplot as plt

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

def load_model():
    model_folder_path = '/home/ubuntu/efs-w210-capstone-ebs/04B.Local_Model_Files/20240713_Non_Overlapping_Consl_180_FINAL'
    trained_model_file = f"{model_folder_path}/20240712_Transformers_Non_Overlapping_run_143_of_180.pt"
    model = TransformerModel(input_size=55, hidden_size=64, num_layers=2, output_size=1, dropout=0.3)
    model.load_state_dict(torch.load(trained_model_file))
    model.eval()
    return model

# Autogen setup
class TrackableAssistantAgent(autogen.AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        unique_message = (sender.name, message.get('content', '') if isinstance(message, dict) else message)
        if unique_message not in st.session_state['seen_messages']:
            st.session_state['seen_messages'].add(unique_message)
            if st.session_state['show_chat_content']:
                with st.chat_message(sender.name):
                    if isinstance(message, dict) and 'content' in message:
                        try:
                            json_content = json.loads(message['content'])
                            formatted_content = json.dumps(json_content, indent=2)
                            st.markdown(f"**{sender.name}**:\n```json\n{formatted_content}\n```")
                        except json.JSONDecodeError:
                            st.markdown(f"**{sender.name}**:\n{message['content']}")
                    else:
                        st.markdown(f"**{sender.name}**:\n{message}")
        return super()._process_received_message(message, sender, silent)

class TrackableUserProxyAgent(autogen.UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        unique_message = (sender.name, message.get('content', '') if isinstance(message, dict) else message)
        if unique_message not in st.session_state['seen_messages']:
            st.session_state['seen_messages'].add(unique_message)
            if st.session_state['show_chat_content']:
                with st.chat_message(sender.name):
                    if isinstance(message, dict) and 'content' in message:
                        try:
                            json_content = json.loads(message['content'])
                            formatted_content = json.dumps(json_content, indent=2)
                            st.markdown(f"**{sender.name}**:\n```json\n{formatted_content}\n```")
                        except json.JSONDecodeError:
                            st.markdown(f"**{sender.name}**:\n{message['content']}")
                    else:
                        st.markdown(f"**{sender.name}**:\n{message}")
        return super()._process_received_message(message, sender, silent)

def send_log_data_and_get_model_results(start_time: str, num_tests: int) -> Union[dict, str]:
    print("Loading the model...")
    model = load_model()
    print("Model loaded.")
    
    data_dir = '/home/ubuntu/efs-w210-capstone-ebs/04A.Local_Data_Files'
    save_dir = '/home/ubuntu/efs-w210-capstone-ebs/04C.Local_Inference_Eval_Files'
    file_prefix = "06.20240714_062624_non_overlap_full_test"

    print("Loading test data...")
    X_test = np.load(f"{data_dir}/{file_prefix}_X_test.npy")
    y_test = np.load(f"{data_dir}/{file_prefix}_y_test.npy")

    input_file = f"{data_dir}/03.20240715_143154_orig_input_w_seq_info_FINAL.parquet"
    original_df = pd.read_parquet(input_file)
    original_df['Seq_Num'] = original_df['Seq_Num'].astype(int)

    print(f"Filtering test dataset based on start time: {start_time}")
    filtered_df = original_df[(original_df['Train_Test'] == 'Test') & (original_df['time_start'] == start_time)]
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

    start_seq_num = filtered_df['Seq_Num'].values[0]
    
    num_records_per_test = 33
    max_end_seq_num = original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].max()
    expected_end_seq_num = start_seq_num + num_tests * num_records_per_test - 1

    if expected_end_seq_num > max_end_seq_num:
        max_allowed_runs = (max_end_seq_num - start_seq_num + 1) // num_records_per_test
        num_tests = max_allowed_runs
        expected_end_seq_num = start_seq_num + num_tests * num_records_per_test - 1

    start_index_x_test = (start_seq_num - original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].min()) // num_records_per_test
    end_index_x_test = min(start_index_x_test + num_tests, len(X_test))

    X_test_limited = X_test[start_index_x_test:end_index_x_test]
    y_test_limited = y_test[start_index_x_test:end_index_x_test]

    X_test_limited = X_test_limited[:, :, 1:]  # Drop the first column
    X_test_tensor = torch.tensor(X_test_limited, dtype=torch.float32)

    print("Making predictions...")
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_outputs_sigmoid = torch.sigmoid(test_outputs)
        predictions = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).flatten()

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

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    final_file = f"{save_dir}/03B.{timestamp}_agent1_non_overlap_model2_consl.csv"
    tracking_df.to_csv(final_file, index=False)

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

    print("Returning results.")
    return results

def create_interaction_diagram():
    G = nx.DiGraph()
    G.add_edges_from([
        ("user_proxy", "chat_manager"),
        ("chat_manager", "log_data_retriever"),
        ("log_data_retriever", "chat_manager"),
        ("chat_manager", "user_proxy"),
        ("chat_manager", "log_data_writer"),
        ("log_data_writer", "chat_manager"),
    ])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color='black', linewidths=1, font_size=15, arrowsize=20)
    image_path = '/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/09.Images_Docs_Other_Dynamic_Content/interaction_diagram.png'
    plt.savefig(image_path)
    return image_path

def main():
    st.set_page_config(page_title="GabbleGrid", page_icon="🔍", layout="wide")
    st.title('GabbleGrid')

    if 'show_chat_content' not in st.session_state:
        st.session_state['show_chat_content'] = False

    if 'seen_messages' not in st.session_state:
        st.session_state['seen_messages'] = set()

    with st.sidebar:
        st.header("OpenAI Configuration")
        selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'], index=1)
        api_key = "sk-proj-iQtcgUJOOf4n53Bs6uyqT3BlbkFJnEIqUeEwXjbjVMcDVqiz"

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input('Start Time YYYY-MM-DD HH:MM:SS', key='start_time_input')
    with col2:
        num_tests = st.number_input('Number of Tests', min_value=1, value=10, key='num_tests_input')

    if st.button('Run Inference'):
        st.info('<placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text  >')
        time.sleep(2)
        st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/10.Image_and_Static_Content/IMG_0252.jpeg')

        if not api_key:
            st.warning('Please provide a valid OpenAI API key', icon="⚠️")
            st.stop()

        llm_config = {
            "config_list": [
                {
                    "model": selected_model,
                    "api_key": api_key
                }
            ]
        }

        user_proxy = TrackableUserProxyAgent(
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
            llm_config=llm_config
        )

        log_data_retriever = TrackableAssistantAgent(
            name="log_data_retriever",
            system_message="For all tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
            llm_config=llm_config,
        )

        log_data_writer = TrackableAssistantAgent(
            name="log_data_writer",
            system_message="""
            You are a creative writer and your job is to take the data extracted by log_data_retriever and 
            summarize the content in a email with subject header and body. Please keep the email concise and focus on the analysis of the confusion matrix.
            Specifically since this is a log anomaly detection task, please try to analyze the recall and precision of Class 1 (ie Alert)
            """,
            llm_config=llm_config,
        )

        user_proxy.register_for_execution()(send_log_data_and_get_model_results)
        log_data_retriever.register_for_llm(description="Run inference on test data and evaluate model performance")(send_log_data_and_get_model_results)

        groupchat = autogen.GroupChat(agents=[user_proxy, log_data_retriever, log_data_writer], messages=[], max_round=10)
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        if 'chat_initiated' not in st.session_state:
            st.session_state.chat_initiated = False

        if not st.session_state.chat_initiated:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run_chat():
                task = f"""
                Your task is to simply run the only function you have access to and return the content generated. Please use these two parameter values:
                start_time = '{start_time}'
                num_tests = {num_tests}
                In case the start_time does not exist, the function will return a list of other available times. Please select the nearest one and proceed with the task.
                """
                
                with Cache.disk() as cache:
                    await user_proxy.a_initiate_chat(
                        manager, message=task, summary_method="reflection_with_llm", cache=cache
                    )

            loop.run_until_complete(run_chat())
            loop.close()

            st.session_state.chat_initiated = True

        results = None
        for message in user_proxy.chat_messages[manager]:
            if isinstance(message, dict) and 'content' in message:
                try:
                    content = message['content']
                    if isinstance(content, str) and content.startswith('{') and content.endswith('}'):
                        results = json.loads(content)
                        if isinstance(results, dict) and 'metrics' in results:
                            break
                except Exception as e:
                    st.error(f"Error parsing message content: {e}")
                    continue

        if results is None:
            st.error("Failed to get results from the model")
        else:
            st.subheader('Outcome')
            final_event = None
            for message in reversed(user_proxy.chat_messages[manager]):
                if isinstance(message, dict) and message.get('name') == 'log_data_writer':
                    final_event = message.get('content', '')
                    break
            if final_event:
                st.markdown(final_event)
            else:
                st.write("No Outcome found.")

            st.subheader('Metrics')
            st.json(results['metrics'])
            
            st.subheader('Confusion Matrix')
            st.table(results['confusion_matrix'])
            
            st.subheader('Agents & Interaction')
            conversation_transcript = []
            
            for message in user_proxy.chat_messages[manager]:
                if isinstance(message, dict):
                    speaker = message.get('name', 'Unknown')
                    if speaker == 'Unknown':
                        speaker = 'user_proxy'
                    content = message.get('content', '')
                    
                    if not content and 'tool_calls' in message:
                        tool_calls = message['tool_calls']
                        for call in tool_calls:
                            if call['type'] == 'function':
                                function_name = call['function']['name']
                                arguments = call['function']['arguments']
                                content = f"***** Suggested tool call: {function_name} *****\nArguments:\n{arguments}\n****************************************************************************************************"
                    
                    conversation_transcript.append("--------------------------------------------------------------------------------")
                    conversation_transcript.append(f"<span style='color: blue;'>{speaker} (to chat_manager):</span>\n\n{content}")
            
            conversation_transcript.append("--------------------------------------------------------------------------------")
            st.markdown('\n\n'.join(conversation_transcript), unsafe_allow_html=True)
            
            # Display interaction diagram
            st.subheader("Diagram 1: Initial Interaction Diagram")
            st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/10.Image_and_Static_Content/IMG_0252.jpeg')
            
            # Create and display new interaction diagram
            st.subheader("Diagram 2: Interaction Steps Diagram")
            interaction_diagram_path = create_interaction_diagram()
            st.image(interaction_diagram_path)

if __name__ == '__main__':
    main()