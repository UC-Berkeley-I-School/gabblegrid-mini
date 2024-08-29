# playground_log_inference.py

import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime
from autogen.cache import Cache
import autogen
from .utils.autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent  # Updated import
from .utils.inference import send_log_data_and_get_model_results  # Updated import
from .utils.plotting import create_interaction_diagram  # Updated import
from utils.sidebar_utils import csv_path, input_file

def initialize_log_agents(llm_config):
    user_proxy = TrackableUserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "group", "use_docker": False},
        max_consecutive_auto_reply=10,
        system_message="""Reply TERMINATE if the task been solved at full satisfaction. Otherwise, reply CONTINUE or the reason why the task is not solved yet.""",
        llm_config=llm_config
    )

    log_data_retriever = TrackableAssistantAgent(
        name="log_data_retriever",
        system_message=f"For all tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. For finding the model tracking file, please use the csv_path '{csv_path}'",
        llm_config=llm_config,
    )
    
    log_data_writer = TrackableAssistantAgent(
        name="log_data_writer",
        system_message="""
        You are a creative writer and your job is to take the data extracted by log_data_retriever and 
        summarize the content in an email with subject header and body. Please keep the email concise and focus on the analysis of the confusion matrix.
        Specifically since this is a log anomaly detection task, please try to analyze the recall and precision of Class 1 (ie Alert).
        In producing the content, please do not use markdown headings like # or ## and so, please limit the formatting to bold and italics only.
        Finally, please generate your content in response to a specific task, please generate the content just one and never more than once.
        """,
        llm_config=llm_config,
    )

    return {"user_proxy": user_proxy, "log_data_retriever": log_data_retriever, "log_data_writer": log_data_writer}

def register_log_functions(agents):
    agents["user_proxy"].register_for_execution()(send_log_data_and_get_model_results)
    agents["log_data_retriever"].register_for_llm(description="Run inference on test data and evaluate model performance")(send_log_data_and_get_model_results)

def setup_log_group_chat(agents, llm_config):
    groupchat = autogen.GroupChat(agents=list(agents.values()), messages=[], max_round=10)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return groupchat, manager


async def run_chat_log_agents(user_proxy, manager, start_time, inference_params):
    num_tests = inference_params["num_tests"]
    
    # Adjust prompt based on num_tests
    if num_tests < 50:
        additional_context = "Note: The number of tests is relatively small, so please ensure that the results are representative."
    else:
        additional_context = ""

    task = f"""
    Your task is to simply run the only function you have access to and return the content generated. Please use these parameter values:
    start_time = '{start_time}'
    num_tests = {num_tests}
    input_length = {inference_params["input_length"]}
    gap = {inference_params["gap"]}
    prediction_period = {inference_params["prediction_period"]}
    max_events = {inference_params["max_events"]}
    {additional_context}
    In case the start_time does not exist, the function will return a list of other available times. Please select the nearest one and proceed with the task.
    Finally, in responding to a task, please generate your content just one and never more than once.
    """
    
    # st.write(f"Task being sent to user_proxy:\n{task}")  # Debug: log the task being sent

    try:
        with Cache.disk() as cache:
            await user_proxy.a_initiate_chat(
                manager, message=task, summary_method="reflection_with_llm", cache=cache
            )
    except Exception as e:
        st.error(f"Unexpected error during chat initiation: {e}")
        return None

    for message in user_proxy.chat_messages[manager]:
        if isinstance(message, dict) and 'content' in message:
            try:
                content = message['content']
                # st.write(f"Received message content: {content}")  # Debug: log the content received
                if isinstance(content, str):
                    if content.startswith('{') and content.endswith('}'):
                        results = json.loads(content)
                        if isinstance(results, dict):
                            if 'metrics' in results:
                                return results
                            elif 'error' in results:
                                return results  # Return the error message
                    elif "No matching model found" in content:
                        return {"error": content}  # Return the error as a dict
            except Exception as e:
                st.error(f"Error parsing message content: {e}")
    
    return None


async def run_log_inference(model_config, inference_params, openai_api_key):
    # st.write("API Key received:", api_key)  # Debug print
    error_messages = []

    if not openai_api_key:
        st.warning('Please provide a valid OpenAI API key', icon="⚠️")
        return

    llm_config = {
        "config_list": [
            {
                "model": model_config["model"],
                "api_key": openai_api_key,  # Use the api_key from parameter
                "temperature": model_config["temperature"],
                "max_tokens": model_config["max_tokens"],
                "top_p": model_config["top_p"]
            }
        ]
    }

    try:
        agents = initialize_log_agents(llm_config)
    except Exception as e:
        error_messages.append(f"Error initializing agents: {e}")
        return display_errors(error_messages)

    try:
        register_log_functions(agents)
    except Exception as e:
        error_messages.append(f"Error registering functions: {e}")
        return display_errors(error_messages)

    try:
        groupchat, manager = setup_log_group_chat(agents, llm_config)
    except Exception as e:
        error_messages.append(f"Error setting up group chat: {e}")
        return display_errors(error_messages)

    start_time = datetime.combine(inference_params["selected_date"], inference_params["selected_time"])

    try:
        results = await run_chat_log_agents(agents["user_proxy"], manager, start_time, inference_params)
    except Exception as e:
        error_messages.append(f"Error during inference: {e}")
        return display_errors(error_messages)

    if isinstance(results, dict):
        if 'error' in results:
            return display_errors([results['error']])
        else:
            display_log_results(results, agents["user_proxy"].chat_messages[manager])
            return results
    else:
        return display_errors(["Failed to get results from the model or unexpected result type."])

def display_errors(error_messages):
    st.markdown("<h2 style='color: tomato;'>Errors</h2>", unsafe_allow_html=True)
    st.markdown('<br>'.join(error_messages), unsafe_allow_html=True)



def display_log_results(results, chat_messages):
    # st.markdown("<h3 class='custom-tomato' style='color: tomato;'>All Content is generated by teams of Agents working together on a task</h3>", unsafe_allow_html=True)
    # st.markdown("<h4 class='custom-grey' style='color: grey;'>In this example, we will see 3 teams in actions starting with </h4>", unsafe_allow_html=True)
    st.markdown("<h3 class='custom-teal' style='color: teal;'>Team 1: Generate Email with key Log Metrics Only </h3>", unsafe_allow_html=True)
    st.markdown("<p class='custom-cerulean' style='color: cerulean;'><em>This is the final output produced by one of the agents in the group. For details of the process please see the section 'Agents & Interaction'.</em></p>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    
    final_event = next((message.get('content', '') for message in reversed(chat_messages) if isinstance(message, dict) and message.get('name') == 'log_data_writer'), None)
    if final_event:
        st.markdown(final_event)
    else:
        st.write("No Outcome found.")

    st.markdown('<hr>', unsafe_allow_html=True)
    
    display_metrics(results)
    display_agent_interactions(chat_messages)

    st.markdown("<h5 style='color: grey;'>Interaction Map</h5>", unsafe_allow_html=True)
    # st.subheader("Interaction Map")
    interaction_diagram_path = create_interaction_diagram()
    st.image(interaction_diagram_path)

def display_metrics(results):
    st.markdown("<h5 style='color: grey;'>Other Content</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: grey;'>Metrics</h5>", unsafe_allow_html=True)
    # st.subheader('Metrics')
    # st.subheader('Metrics')
    metrics = results['metrics']
    formatted_metrics = "\n".join([
        f"Accuracy: {metrics['Accuracy']:.4f}",
        f"Precision (Class 1): {metrics['Precision (Class 1)']:.4f}",
        f"Recall (Class 1): {metrics['Recall (Class 1)']:.4f}",
        f"F1 Score: {metrics['F1 Score']:.4f}",
        f"True Positives: {metrics['True Positives']}",
        f"False Positives: {metrics['False Positives']}",
        f"True Negatives: {metrics['True Negatives']}",
        f"False Negatives: {metrics['False Negatives']}"
    ])
    st.text(formatted_metrics)

    confusion_matrix = results['confusion_matrix']
    conf_matrix_df = pd.DataFrame(
        confusion_matrix,
        index=["Actual - Normal", "Actual - Alert"],
        columns=["Predicted - Normal", "Predicted - Alert"]
    )
    st.markdown("**Confusion Matrix**")
    st.dataframe(conf_matrix_df)

def display_agent_interactions(chat_messages):
    # st.subheader('Agents & Interaction')
    st.markdown("<h5 style='color: grey;'>Agents & Interaction</h5>", unsafe_allow_html=True)
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