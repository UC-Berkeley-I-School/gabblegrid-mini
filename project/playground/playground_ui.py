import streamlit as st
import os
# from .playground_config import MODELS, DEFAULT_API_KEY, ESSENTIAL_READING_FOLDER
from .playground_utils import display_pdf_as_images
from .playground_text import key_parameters  # Import key_parameters from playground_text
from utils.sidebar_utils import render_sidebar#, render_essential_reading
from utils.sidebar_utils import MODELS, ESSENTIAL_READING_FOLDER


def render_main_content(min_time, max_time):
    st.markdown("<h2 style='color: grey;'>Parameters</h2>", unsafe_allow_html=True)
    
    # Parameters Explained section
    with st.expander("Parameters Explained"):
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Fundamentals_08.png', use_column_width=True)
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Fundamentals_06_B.png', use_column_width=True)  # Add the second image
        st.markdown(key_parameters, unsafe_allow_html=True)  # Add key_parameters content

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_date = st.date_input('Start Date', min_value=min_time.date(), max_value=max_time.date(), value=min_time.date(), key='start_date_input')
    with col2:
        selected_time = st.time_input('Start Time', value=min_time.time(), key='start_time_input')
    with col3:
        st.text_input('Sliding Window', value='Sequential', disabled=True, key='sliding_window_input')

    col4, col5, col6 = st.columns(3)
    with col4:
        st.text_input('AutoPicker', value='Manual', disabled=True, key='auto_picker_input')
    with col5:
        max_events = st.number_input('Max Events', min_value=1, value=30, key='max_events_input')
    with col6:
        num_tests = st.number_input('Number of Samples', min_value=1, value=10, key='num_tests_input')

    col7, col8, col9 = st.columns(3)
    with col7:
        input_length = st.number_input('Observation Period', min_value=1, value=30, key='input_length_input')
    with col8:
        gap = st.number_input('Gap Period', min_value=1, value=2, key='gap_input')
    with col9:
        prediction_period = st.number_input('Prediction Period', min_value=1, value=1, key='prediction_period_input')

    return {
        "selected_date": selected_date,
        "selected_time": selected_time,
        "input_length": input_length,
        "num_tests": num_tests,
        "gap": gap,
        "prediction_period": prediction_period,
        "max_events": max_events
    }