import streamlit as st
import pandas as pd
from datetime import datetime
from .playground_text import key_parameters, playground_intro, playground_intro_expanded

def load_templates():
    try:
        df = pd.read_csv('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/admin/20240802_experiment_templates.csv')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def render_main_content(min_time, max_time):
    # st.markdown("<h4 style='color: grey;'>Parameters</h4>", unsafe_allow_html=True)
    
    with st.expander("Parameters Explained"):
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Fundamentals_08.png', use_column_width=True)
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Fundamentals_06_B.png', use_column_width=True)
        st.markdown(key_parameters, unsafe_allow_html=True)

    templates = load_templates()
    templates['ID_Description'] = templates['ID'] + ' - ' + templates['Description'].fillna('')
    template_options = ["No template"] + templates['ID_Description'].tolist()
    selected_template = st.selectbox("Select a template or enter parameters manually", template_options)

    if selected_template != "No template":
        template_id = selected_template.split(' - ')[0]
        template_data = templates[templates['ID'] == template_id].iloc[0]
        disabled = True
    else:
        template_data = None
        disabled = False

    max_events_options = [5, 10, 20, 30, 40, 50]
    input_length_options = [20, 30]
    gap_options = [1, 2, 3, 4, 5]
    prediction_period_options = [1]

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_date = st.date_input('Start Date', 
                                      min_value=min_time.date(), 
                                      max_value=max_time.date(), 
                                      value=pd.to_datetime(template_data['Start Date']).date() if template_data is not None else min_time.date(),
                                      disabled=disabled,
                                      key='start_date_input')
    with col2:
        selected_time = st.time_input('Start Time', 
                                      value=pd.to_datetime(template_data['Start Time']).time() if template_data is not None else min_time.time(),
                                      disabled=disabled,
                                      key='start_time_input')
    with col3:
        st.text_input('Sliding Window', 
                      value=template_data['Sliding Window'] if template_data is not None else 'Sequential',
                      disabled=True,
                      key='sliding_window_input')

    col4, col5, col6 = st.columns(3)
    with col4:
        st.text_input('AutoPicker', 
                      value=template_data['AutoPicker'] if template_data is not None else 'Manual',
                      disabled=True,
                      key='auto_picker_input')
    with col5:
        max_events = st.selectbox('Max Events', 
                                  options=max_events_options,
                                  index=max_events_options.index(int(template_data['Max Events'])) if template_data is not None else 0,
                                  disabled=disabled,
                                  key='max_events_input')
    with col6:
        num_tests = st.number_input('Number of Samples', 
                                    min_value=1, 
                                    value=int(template_data['Number of Samples']) if template_data is not None else 10,
                                    disabled=disabled,
                                    key='num_tests_input')

    col7, col8, col9 = st.columns(3)
    with col7:
        input_length = st.selectbox('Observation Period', 
                                    options=input_length_options,
                                    index=input_length_options.index(int(template_data['Observation Period'])) if template_data is not None else 0,
                                    disabled=disabled,
                                    key='input_length_input')
    with col8:
        gap = st.selectbox('Gap Period', 
                           options=gap_options,
                           index=gap_options.index(int(template_data['Gap Period'])) if template_data is not None else 0,
                           disabled=disabled,
                           key='gap_input')
    with col9:
        prediction_period = st.selectbox('Prediction Period', 
                                         options=prediction_period_options,
                                         index=prediction_period_options.index(int(template_data['Prediction Period'])) if template_data is not None else 0,
                                         disabled=disabled,
                                         key='prediction_period_input')

    return {
        "selected_date": selected_date,
        "selected_time": selected_time,
        "input_length": input_length,
        "num_tests": num_tests,
        "gap": gap,
        "prediction_period": prediction_period,
        "max_events": max_events
    }

def render_agent_team_selection():
    # st.markdown("<h4 style='color: grey;'>Select Teams with specific goals</h4>", unsafe_allow_html=True)

    team_selection = st.multiselect(
        # "Select Teams to Run Inference",
        "",
        options=[
            "Team 1: Basic Log Inference",
            "Team 2: Current Weather and Search",
            "Team 3: Historical Weather Trends"
        ],
        default=[
            "Team 1: Basic Log Inference",
            "Team 2: Current Weather and Search",
            "Team 3: Historical Weather Trends"
        ]
    )

    return team_selection