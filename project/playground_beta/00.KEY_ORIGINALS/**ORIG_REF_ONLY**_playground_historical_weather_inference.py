import streamlit as st
import asyncio
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from autogen.cache import Cache
import autogen
from .beta_utils.autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import re
import os
from typing import Annotated

data_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/02.Local_Data_Files'
save_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/admin/01.Templates'
file_prefix = "06.20240714_062624_non_overlap_full_test"

def initialize_historical_weather_agents(llm_config):
    user_proxy = TrackableUserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "group", "use_docker": False},
        max_consecutive_auto_reply=10,
        system_message="""Reply TERMINATE if the task has been solved to full satisfaction. Otherwise, reply CONTINUE or the reason why the task is not solved yet.""",
        llm_config=llm_config
    )

    log_historical_weather_retriever = TrackableAssistantAgent(
        name="log_historical_weather_retriever",
        system_message="""You are an agent specialized in retrieving and processing historical weather data for log analysis. Use the functions provided to fetch and analyze weather information. Reply TERMINATE when the task is done.""",
        llm_config=llm_config
    )

    log_historical_weather_writer = TrackableAssistantAgent(
        name="log_historical_weather_writer",
        system_message="""
        You are a creative writer and your job is to take the data extracted by log_historical_weather_retriever and 
        summarize the content in an email with subject header and body. Please keep the email concise and focus on the general analysis of the data that you receive.
        Specifically, since this is a log anomaly detection task, please try to analyze the data in that context
        In producing the content, please do not use markdown headings like # or ##, and please limit the formatting to bold and italics only.
        Finally, please generate your content in response to a specific task, please generate the content just once and never more than once.
        """,
        llm_config=llm_config
    )

    return {"user_proxy": user_proxy, "log_historical_weather_retriever": log_historical_weather_retriever, "log_historical_weather_writer": log_historical_weather_writer}

def prepare_data_for_model(X_test, start_seq_num, num_records_per_test, num_tests, original_df, max_events):
    start_index_x_test = (start_seq_num - original_df[original_df['Train_Test'] == 'Test']['Seq_Num'].min()) // num_records_per_test
    end_index_x_test = min(start_index_x_test + num_tests, len(X_test))
    X_test_limited = X_test[start_index_x_test:end_index_x_test]
    X_test_limited = X_test_limited[:, :, 1:max_events + 16]
    X_test_tensor = torch.tensor(X_test_limited, dtype=torch.float32)
    return X_test_tensor, start_index_x_test, end_index_x_test

def register_historical_weather_functions(agents):
    @agents["user_proxy"].register_for_execution()
    @agents["log_historical_weather_retriever"].register_for_llm(description="Retrieve and process historical weather data")
    def get_historical_weather_data(
        max_events: Annotated[int, "The maximum number of events"],
        input_length: Annotated[int, "The length of the input sequence"],
        gap: Annotated[int, "The gap between sequences"],
        prediction_period: Annotated[int, "The prediction period"],
        selected_date: Annotated[str, "The selected date for retrieving historical weather data"],
        selected_time: Annotated[str, "The selected time for retrieving historical weather data"],
        num_tests: Annotated[int, "The number of tests to run"]
    ) -> dict:
        start_time = f"{selected_date} {selected_time}"
    
        try:
            # Load test data
            X_test = np.load(f"{data_dir}/{file_prefix}_X_test.npy")
            y_test = np.load(f"{data_dir}/{file_prefix}_y_test.npy")

            # Load original parquet file
            input_file = f"{data_dir}/03.20240715_143154_orig_input_w_seq_info_FINAL.parquet"
            original_df = pd.read_parquet(input_file)
            original_df['Seq_Num'] = original_df['Seq_Num'].astype(int)

            # Filter data based on start_time
            filtered_df = original_df[(original_df['Train_Test'] == 'Test') & (original_df['time_start'] == start_time)]
            if filtered_df.empty:
                all_times_df = original_df[original_df['Train_Test'] == 'Test'].copy()
                all_times_df['time_start'] = pd.to_datetime(all_times_df['time_start'])
                nearest_time = all_times_df.iloc[(all_times_df['time_start'] - pd.to_datetime(start_time)).abs().argsort()[:1]]['time_start'].values[0]
                start_time = nearest_time  # Update the start_time to the nearest available time
                filtered_df = original_df[(original_df['Train_Test'] == 'Test') & (original_df['time_start'] == start_time)].copy()

            if filtered_df.empty:
                return {"error": f"No data found for start_time: {start_time}"}

            start_seq_num = int(filtered_df['Seq_Num'].values[0])
            num_records_per_test = input_length + gap + prediction_period

            # Prepare data for the model
            X_test_tensor, start_index_x_test, end_index_x_test = prepare_data_for_model(X_test, start_seq_num, num_records_per_test, num_tests, original_df, max_events)

            # Here you would typically run your model inference
            # For this example, we'll just use random predictions
            predictions = np.random.randint(0, 2, size=(end_index_x_test - start_index_x_test))

            # Calculate metrics
            conf_matrix = confusion_matrix(y_test[start_index_x_test:end_index_x_test], predictions, labels=[0, 1])
            precision = precision_score(y_test[start_index_x_test:end_index_x_test], predictions, zero_division=0)
            recall = recall_score(y_test[start_index_x_test:end_index_x_test], predictions, zero_division=0)
            accuracy = accuracy_score(y_test[start_index_x_test:end_index_x_test], predictions)
            f1 = f1_score(y_test[start_index_x_test:end_index_x_test], predictions, zero_division=0)

            tn, fp, fn, tp = conf_matrix.ravel()

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            metrics_data = {
                'Experiment': [timestamp] * num_tests,
                'Sample': list(range(1, num_tests + 1)),
                'Max_Events': [max_events] * num_tests,
                'Input_Length': [input_length] * num_tests,
                'Gap': [gap] * num_tests,
                'Prediction_Period': [prediction_period] * num_tests,
                'Exp_Start_Time': [start_time] * num_tests,
                'Num_Tests': [num_tests] * num_tests,
                'Model_Name': ['random_model'] * num_tests,
                'Precision': [precision] * num_tests,
                'Recall': [recall] * num_tests,
                'Accuracy': [accuracy] * num_tests,
                'F1_Score': [f1] * num_tests,
                'TN': [tn] * num_tests,
                'FP': [fp] * num_tests,
                'FN': [fn] * num_tests,
                'TP': [tp] * num_tests
            }
            metrics_df = pd.DataFrame(metrics_data)

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

            combined_df = pd.concat([metrics_df, tracking_df], axis=1)

            master_tracking_file = f"{data_dir}/03B.20240716_072206_orig_parquet_mapper_agents.parquet"
            master_tracking_df = pd.read_parquet(master_tracking_file)

            merged_df = combined_df.merge(master_tracking_df, left_on='Source_Seq_Start', right_on='Seq_Num', how='left')
            merged_df.rename(columns={'time_start_y': 'Sample_Start_Time'}, inplace=True)

            columns_to_keep = [
                "Experiment", "Sample", "Max_Events", "Input_Length", "Gap", "Prediction_Period",
                "Exp_Start_Time", "Num_Tests", "Model_Name", "Precision", "Recall", "Accuracy", "F1_Score", "TN", "FP", "FN", "TP",
                "Total_Seq_Start", "Total_Seq_End", "Source_Seq_Start", "Source_Seq_End", "Gap_Seq_Start",
                "Gap_Seq_End", "Prediction_Seq_Start", "Prediction_Seq_End", "Predicted", "Actual", "Sample_Start_Time",
                "Class", "unique_events", "most_frequent_event", "transitions", "entropy", "hour_of_day", "day_of_week",
                "event_count", "top_event_frequency", "prev_event_count", "transition_rate", "high_transition_rate",
                "prev_entropy", "entropy_change", "rolling_event_count", "rolling_unique_event_count"
            ]

            filtered_df = merged_df[columns_to_keep].copy()

            eventid_encoding_file = f"{data_dir}/08.20240716031626_event_ID_int_template_mapping.csv"
            eventid_encoding_df = pd.read_csv(eventid_encoding_file)
            eventid_to_template = dict(zip(eventid_encoding_df['EncodedValue'], eventid_encoding_df['EventTemplate']))

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

            filtered_df['runtime_most_frequent_consl_text'] = consolidate_events_to_text(filtered_df, 'Source_Seq_Start', 'Source_Seq_End', 'most_frequent_event', master_tracking_df, eventid_to_template)
            filtered_df['runtime_least_frequent_consl_text'] = consolidate_events_to_text(filtered_df, 'Source_Seq_Start', 'Source_Seq_End', 'most_frequent_event', master_tracking_df, eventid_to_template)

            final_file = f"{save_dir}/A-Template_Detail.parquet"
            filtered_df['Experiment'] = filtered_df['Experiment'].astype(str)
            filtered_df['Exp_Start_Time'] = filtered_df['Exp_Start_Time'].astype(str)  # Convert Exp_Start_Time to string

            if os.path.exists(final_file):
                existing_df = pd.read_parquet(final_file)
                combined_final_df = pd.concat([existing_df, filtered_df])
                combined_final_df.to_parquet(final_file, index=False)
            else:
                filtered_df.to_parquet(final_file, index=False)

            # Convert the DataFrame to a dictionary, ensuring Timestamps are converted to strings
            return filtered_df.tail(2).applymap(str).to_dict(orient='records')

        except Exception as e:
            return {"error": str(e)}

def setup_weather_group_chat(agents, llm_config):
    groupchat = autogen.GroupChat(agents=list(agents.values()), messages=[], max_round=5)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return groupchat, manager

async def run_chat_weather_agents(user_proxy, manager, inference_params):
    task = f"""
    Your task is to run the get_historical_weather_data function with these parameters:
    max_events = {inference_params["max_events"]}
    input_length = {inference_params["input_length"]}
    gap = {inference_params["gap"]}
    prediction_period = {inference_params["prediction_period"]}
    selected_date = '{inference_params["selected_date"]}'
    selected_time = '{inference_params["selected_time"]}'
    num_tests = {inference_params["num_tests"]}
    After running the function, please summarize the results in a concise email format.
    """

    try:
        with Cache.disk() as cache:
            await user_proxy.a_initiate_chat(
                manager, message=task, summary_method="reflection_with_llm", cache=cache
            )
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

    # Find the last message from log_historical_weather_writer
    writer_message = next((msg for msg in reversed(user_proxy.chat_messages[manager]) 
                           if isinstance(msg, dict) and msg.get('name') == 'log_historical_weather_writer'), None)
    
    if writer_message and 'content' in writer_message:
        return writer_message['content']
    else:
        return "No summary generated by log_historical_weather_writer"



async def run_historical_weather_inference(model_config, inference_params):
    error_messages = []

    if not model_config["api_key"]:
        st.warning('Please provide a valid OpenAI API key', icon="⚠️")
        return

    llm_config = {
        "config_list": [
            {
                "model": model_config["model"],
                "api_key": model_config["api_key"],
                "temperature": model_config["temperature"],
                "max_tokens": model_config["max_tokens"],
                "top_p": model_config["top_p"]
            }
        ]
    }

    try:
        agents = initialize_historical_weather_agents(llm_config)
    except Exception as e:
        error_messages.append(f"Error initializing agents: {e}")
        return display_errors(error_messages)

    try:
        register_historical_weather_functions(agents)
    except Exception as e:
        error_messages.append(f"Error registering functions: {e}")
        return display_errors(error_messages)

    try:
        groupchat, manager = setup_weather_group_chat(agents, llm_config)
    except Exception as e:
        error_messages.append(f"Error setting up group chat: {e}")
        return display_errors(error_messages)

    try:
        results = await run_chat_weather_agents(agents["user_proxy"], manager, inference_params)
    except Exception as e:
        error_messages.append(f"Error during inference: {e}")
        return display_errors(error_messages)

    if results:
        if isinstance(results, dict) and 'error' in results:
            return display_errors([results['error']])
        else:
            display_weather_results(results)
            display_agent_interactions(agents["user_proxy"].chat_messages[manager])
            return results
    else:
        return display_errors(["Failed to get results from the model"])


def display_errors(error_messages):
    st.markdown("<h2 style='color: tomato;'>Errors</h2>", unsafe_allow_html=True)
    st.markdown('<br>'.join(error_messages), unsafe_allow_html=True)

def display_weather_results(results):
    st.markdown("<h2 style='color: teal;'>Historical Weather Analysis Summary</h2>", unsafe_allow_html=True)
    if isinstance(results, dict):
        if 'content' in results:
            st.markdown(results['content'])
        else:
            for key, value in results.items():
                st.markdown(f"**{key}:** {value}")
    elif isinstance(results, str):
        st.markdown(results)
    elif isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                for key, value in item.items():
                    st.markdown(f"**{key}:** {value}")
            else:
                st.write(item)
    else:
        st.write(results)


def display_agent_interactions(chat_messages):
    st.subheader('Agents & Interaction')
    conversation_transcript = []

    for message in chat_messages:
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