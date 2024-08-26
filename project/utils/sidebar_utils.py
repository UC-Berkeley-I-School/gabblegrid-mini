# File: utils/sidebar_utils.py

import streamlit as st
import os
from playground.playground_utils import display_pdf_as_images

# Import API keys from the private file first, then fallback to the public file
try:
    from .api_keys_private import openai_api_key as private_openai_api_key, openweather_api_key as private_openweather_api_key, bing_api_key as private_bing_api_key
except ImportError:
    private_openai_api_key = None
    private_openweather_api_key = None
    private_bing_api_key = None

from .api_keys_public import openai_api_key as public_openai_api_key, openweather_api_key as public_openweather_api_key, bing_api_key as public_bing_api_key

# Determine which keys to use (prefer private over public)
openai_api_key = private_openai_api_key or public_openai_api_key
openweather_api_key = private_openweather_api_key or public_openweather_api_key
bing_api_key = private_bing_api_key or public_bing_api_key

# Set the instance type ('dev' or 'prod')
# instance = 'prod'  # Change to 'dev' or 'prod' as needed
instance = 'dev'  # Change to 'dev' or 'prod' as needed

csv_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv'
input_file = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/02.Local_Data_Files/03.20240715_143154_orig_input_w_seq_info_FINAL.parquet"
MODELS = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o']
ESSENTIAL_READING_FOLDER = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/documents/essential_reading"

######################################## Use hardcoded API keys for testing only in 'dev' ########################################
if instance == 'dev':
    # Use the keys as imported
    pass
else:
    openai_api_key = ""
    openweather_api_key = ""
    bing_api_key = ""
###################################################################################################################################

def render_sidebar(key_suffix=''):
    if 'sidebar_config' not in st.session_state:
        st.session_state.sidebar_config = {}

    # Only render the sidebar for playground
    if key_suffix == 'playground':
        with st.sidebar:
            st.header("Model Configuration")
            st.session_state.sidebar_config["model"] = st.selectbox("Model", MODELS, index=2, key=f'model_selectbox_{key_suffix}')
            
            ######################################## Handle API keys based on instance ########################################
            if 'openai_api_key' not in st.session_state or instance == 'prod':
                st.session_state['openai_api_key'] = openai_api_key
            if 'openweather_api_key' not in st.session_state or instance == 'prod':
                st.session_state['openweather_api_key'] = openweather_api_key
            if 'bing_api_key' not in st.session_state or instance == 'prod':
                st.session_state['bing_api_key'] = bing_api_key
            ###################################################################################################################

            # Display the text inputs with the API keys set from the session state
            st.session_state.sidebar_config["openai_api_key"] = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password", key=f'openai_api_key_input_{key_suffix}')
            st.session_state.sidebar_config["openweather_api_key"] = st.text_input("OpenWeather API Key", value=st.session_state.openweather_api_key, type="password", key=f'openweather_api_key_input_{key_suffix}')
            st.session_state.sidebar_config["bing_api_key"] = st.text_input("Bing API Key", value=st.session_state.bing_api_key, type="password", key=f'bing_api_key_input_{key_suffix}')

            st.session_state.sidebar_config["temperature"] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, key=f'temperature_slider_{key_suffix}')
            st.session_state.sidebar_config["max_tokens"] = st.number_input("Max Tokens", min_value=1, max_value=4096, value=2048, key=f'max_tokens_input_{key_suffix}')
            st.session_state.sidebar_config["top_p"] = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, key=f'top_p_slider_{key_suffix}')
            
            # Update the session state when the API keys are modified in the sidebar
            st.session_state.openai_api_key = st.session_state.sidebar_config["openai_api_key"]
            st.session_state.openweather_api_key = st.session_state.sidebar_config["openweather_api_key"]
            st.session_state.bing_api_key = st.session_state.sidebar_config["bing_api_key"]

            render_essential_reading()
    
    return st.session_state.sidebar_config


def render_essential_reading(key_suffix=''):
    st.header("Essential Reading")
    st.markdown('<div class="essential-reading">', unsafe_allow_html=True)
    if os.path.exists(ESSENTIAL_READING_FOLDER):
        files = sorted([f for f in os.listdir(ESSENTIAL_READING_FOLDER) if not f.startswith('.')])
        if files:
            for idx, file in enumerate(files):
                file_path = os.path.join(ESSENTIAL_READING_FOLDER, file)
                display_name = os.path.splitext(os.path.basename(file))[0]
                with st.expander(display_name):
                    display_pdf_as_images(file_path, display_name, key_prefix=f"essential_reading_{idx}_{key_suffix}")
        else:
            st.markdown("No documents found.")
    else:
        st.markdown("The specified folder does not exist.")
    st.markdown('</div>', unsafe_allow_html=True)
