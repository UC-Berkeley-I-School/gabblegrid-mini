import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import re
import os
from typing import Annotated
from datetime import datetime
from autogen.cache import Cache
import numpy as np
from .playground_historical_weather_display import display_weather_results
from .playground_historical_weather_display import display_agent_interactions
from .playground_historical_weather_display import display_errors
from .agents.historical_weather.agent_initialization import initialize_historical_weather_agents
from .agents.historical_weather.data_preparation import get_latest_experiment_id
from .agents.historical_weather.function_registration import register_historical_weather_functions
from .agents.historical_weather.agent_communication import (
    run_chat_weather_agents,
    generate_plot,
    setup_weather_group_chat,
)
from .agents.historical_weather.data_preparation import (
    prepare_data_for_model,
    consolidate_events_to_text,
)

data_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/02.Local_Data_Files'
save_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/admin/01.Templates'
image_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/playground/01.Experiments/01.Images'
file_prefix = "06.20240714_062624_non_overlap_full_test"
weather_parquet = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/04.Local_Other_Files/20240803_Historical_Weather_94550/openweathermap_livermore.parquet'
experiment_parquet = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/admin/01.Templates/A-Template_Detail.parquet'

async def run_historical_weather_inference(model_config, inference_params, api_key, experiment_id):
    # st.write("API Key received:", api_key)  # Debug print
    error_messages = []

    # Define image_dir before it is used
    image_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/playground/01.Experiments/01.Images'

    if not api_key:
        st.warning('Please provide a valid OpenAI API key', icon="⚠️")
        return

    llm_config = {
        "config_list": [
            {
                "model": model_config["model"],
                "api_key": api_key,  # Use the api_key from parameter
                "temperature": model_config["temperature"],
                "max_tokens": model_config["max_tokens"],
                "top_p": model_config["top_p"]
            }
        ]
    }
    
    try:
        # st.write("Initializing agents...")
        agents = initialize_historical_weather_agents(llm_config)
    except Exception as e:
        error_messages.append(f"Error initializing agents: {e}")
        return display_errors(error_messages)

    try:
        # st.write("Registering functions...")
        register_historical_weather_functions(agents)
    except Exception as e:
        error_messages.append(f"Error registering functions: {e}")
        return display_errors(error_messages)

    try:
        # st.write("Setting up group chat...")
        groupchat, manager = setup_weather_group_chat(agents, llm_config)
    except Exception as e:
        error_messages.append(f"Error setting up group chat: {e}")
        return display_errors(error_messages)

    try:
        # st.write("Running chat with weather agents...")
        results = await run_chat_weather_agents(agents["user_proxy"], manager, inference_params, experiment_id)
        # results = await run_chat_weather_agents(agents["user_proxy"], manager, inference_params, experiment_timestamp)
        # # Debug: Log all messages received during the chat
        # st.write("Debugging messages received during the chat:")
        # for message in agents["user_proxy"].chat_messages[manager]:
        #     if isinstance(message, dict) and 'name' in message:
        #         st.write(f"Message from {message['name']}: {message.get('content', 'No content provided')}")
        #     else:
        #         st.write("Received a message without a 'name' attribute or message structure was unexpected:", message)

    except Exception as e:
        error_messages.append(f"Error during inference: {e}")
        st.error(f"Error during inference: {e}")
        return display_errors(error_messages)

    if not results:
        st.error("No results were returned from the inference process.")
        return display_errors(["Failed to get results from the model"])
    
    if isinstance(results, dict) and 'error' in results:
        return display_errors([results['error']])

    # st.write("Inference results received:", results)

    #################### Part that needs to be changed ###################################
    
    # At this point, results should be valid, so proceed with generating the plot
    try:
        # st.write("Getting latest experiment ID...")
        experiment_id = get_latest_experiment_id(experiment_parquet)
        # st.write(f"Latest experiment ID: {experiment_id}")
    except Exception as e:
        return display_errors([f"Failed to get latest experiment ID: {e}"])


    ###########################################################################################

    plot_params = {
        "experiment_id": experiment_id,
        "image_dir": image_dir  # Ensure this is included
    }
    await generate_plot(agents["user_proxy"], manager, plot_params)

    
    
######### Code change to search based on experiment, not just by latest image ###############

    # Dynamically display the image for the current experiment ID
    try:
        expected_image_file = f"{image_dir}/{experiment_id}_plot.png"
        if os.path.exists(expected_image_file):
            with open(expected_image_file, "rb") as image_file:
                image_bytes = image_file.read()
            st.image(image_bytes, caption=f'Temperature Changes for Experiment {experiment_id}')
        else:
            st.error(f"Plot file not found for Experiment ID: {experiment_id}")
    except Exception as e:
        st.error(f"Error finding or displaying the plot file for Experiment ID {experiment_id}: {e}")

    display_weather_results(results)
    display_agent_interactions(agents["user_proxy"].chat_messages[manager])
    return results