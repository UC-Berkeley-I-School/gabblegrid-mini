# File: utils/sidebar_utils.py

import streamlit as st
import os
from playground.playground_utils import display_pdf_as_images

csv_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv'
input_file = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/02.Local_Data_Files/03.20240715_143154_orig_input_w_seq_info_FINAL.parquet"
MODELS = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o']
ESSENTIAL_READING_FOLDER = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/documents/essential_reading"
BING_API_KEY = "7e534482ac5f46d3a2a979072b19e591"
OPENWEATHER_API_KEY = "864a9bb0b562e1c87e01b38880d5bee7"

def render_sidebar(key_suffix=''):
    if 'sidebar_config' not in st.session_state:
        st.session_state.sidebar_config = {}
    
    # Only render the sidebar for playground
    if key_suffix == 'playground':
        with st.sidebar:
            st.header("Model Configuration")
            st.session_state.sidebar_config["model"] = st.selectbox("Model", MODELS, index=2, key=f'model_selectbox_{key_suffix}')
            
            # Use the existing API key from session state if available
            default_api_key = st.session_state.get("api_key", "")
            st.session_state.sidebar_config["api_key"] = st.text_input("API Key", value=default_api_key, type="password", key=f'api_key_input_{key_suffix}')
            
            st.session_state.sidebar_config["temperature"] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, key=f'temperature_slider_{key_suffix}')
            st.session_state.sidebar_config["max_tokens"] = st.number_input("Max Tokens", min_value=1, max_value=4096, value=2048, key=f'max_tokens_input_{key_suffix}')
            st.session_state.sidebar_config["top_p"] = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, key=f'top_p_slider_{key_suffix}')
            
            # Update the API key in the session state whenever it changes
            st.session_state.api_key = st.session_state.sidebar_config["api_key"]

            render_essential_reading()
    
    # For playground, just update the API key if it exists in the session state
    elif key_suffix == 'playground':
        if 'api_key' in st.session_state:
            st.session_state.sidebar_config["api_key"] = st.session_state.api_key

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
