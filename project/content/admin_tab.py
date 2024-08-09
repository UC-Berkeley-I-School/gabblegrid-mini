import streamlit as st
import pandas as pd
from datetime import datetime
import random
import os

# Temporary block to display the static image
# plot_file_path = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/playground_beta/01.Experiments/01.Images/20240805090256_plot.png"
# if os.path.exists(plot_file_path):
#     with open(plot_file_path, "rb") as image_file:
#         image_bytes = image_file.read()
#     st.image(image_bytes, caption='Temperature Changes')
# else:
#     st.error(f"Plot file not found at {plots_file_path}")


def generate_hex_id():
    return ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

def find_nearest_datetime(df, target_datetime):
    df['time_diff'] = abs(df['time_start'] - target_datetime)
    nearest_row = df.loc[df['time_diff'].idxmin()]
    return nearest_row['time_start'], nearest_row['Seq_Num']

def display_admin_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section h5, .section table, .section td, .section th {
                color: grey;
            }
            .grey-text {
                color: grey;
            }
            .matrix-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Experiment Templates</h2>
            <p>Use this section to create and manage experiment templates with predefined parameter values.</p>
        </div>
    """, unsafe_allow_html=True)


#########

    # # Split into two columns
    # col1, col2 = st.columns([1, 1])

    # with col1:
    #     st.markdown("""
    #         <div class='section'>
    #             <h2>Experiment Templates</h2>
    #             <p>Use this section to create and manage experiment templates with predefined parameter values.</p>
    #         </div>
    #     """, unsafe_allow_html=True)

    # with col2:
    #     st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/20240805_Final_Playground_Beta.mp4')

    # owner = "gaurav.narasimhan@gmail.com"
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    


#########
    
    owner = "gaurav.narasimhan@gmail.com"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load the input file to get the available date range
    input_file = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/02.Local_Data_Files/03.20240715_143154_orig_input_w_seq_info_FINAL.parquet"
    original_df = pd.read_parquet(input_file)
    original_df['time_start'] = pd.to_datetime(original_df['time_start'])
    original_df['Seq_Num'] = original_df['Seq_Num'].astype(int)

    # Filter for only 'Test' data
    test_df = original_df[original_df['Train_Test'] == 'Test']
    min_date = test_df['time_start'].min().date()
    max_date = test_df['time_start'].max().date()

    # Load the master tracker file
    master_tracker = pd.read_csv("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv")

    # Hardcoded unique values for the restricted parameters
    max_events_options = [5, 10, 20, 30, 40, 50]
    input_length_options = [20, 30]
    gap_options = [1, 2, 3, 4, 5]
    prediction_period_options = [1]

    with st.form(key='template_form'):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date, key='admin_start_date')
            sliding_window = st.selectbox('Sliding Window', ['Sequential', 'Non-Sequential'], key='admin_sliding_window')
        
        with col2:
            auto_picker = st.selectbox('AutoPicker', ['Manual', 'Automatic'], key='admin_auto_picker')
            num_tests = st.number_input('Number of Samples', min_value=1, value=10, key='admin_num_tests')
        
        with col3:
            input_length = st.selectbox('Observation Period', options=input_length_options, key='admin_input_length')
            gap = st.selectbox('Gap Period', options=gap_options, key='admin_gap')
        
        with col4:
            prediction_period = st.selectbox('Prediction Period', options=prediction_period_options, key='admin_prediction_period')
            max_events = st.selectbox('Max Events', options=max_events_options, key='admin_max_events')
        
        col5, col6 = st.columns([1, 3])
        with col5:
            start_time = st.time_input('Start Time', key='admin_start_time')
        with col6:
            description = st.text_input('Description', key='admin_description')
        
        submit_button = st.form_submit_button(label='Save Template')

    if submit_button:
        # Check for valid combination
        valid_combination = not master_tracker[(master_tracker['Max Events'] == max_events) & 
                                               (master_tracker['Input Length'] == input_length) & 
                                               (master_tracker['Gap'] == gap) & 
                                               (master_tracker['Prediction Period'] == prediction_period)].empty
        if not valid_combination:
            st.error("Invalid combination of parameters. Please select a valid combination.")
        else:
            # Validate that the selected date and time exist in the Test dataset
            selected_datetime = pd.Timestamp.combine(start_date, start_time)
            if not test_df[test_df['time_start'] == selected_datetime].empty:
                start_seq_num = test_df[test_df['time_start'] == selected_datetime]['Seq_Num'].values[0]
            else:
                nearest_datetime, start_seq_num = find_nearest_datetime(test_df, selected_datetime)
                st.warning(f"Selected date and time not available. Using nearest available: {nearest_datetime}")
                selected_datetime = nearest_datetime
            
            template_id = generate_hex_id()
            new_template = pd.DataFrame({
                'ID': [template_id],
                'Description': [description],
                'Owner': [owner],
                'Timestamp': [timestamp],
                'Start Date': [selected_datetime.strftime("%Y-%m-%d")],
                'Start Time': [selected_datetime.strftime("%H:%M")],
                'Sliding Window': [sliding_window],
                'AutoPicker': [auto_picker],
                'Max Events': [max_events],
                'Number of Samples': [num_tests],
                'Observation Period': [input_length],
                'Gap Period': [gap],
                'Prediction Period': [prediction_period],
                'Start Seq Num': [start_seq_num]
            })
            
            save_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/admin/20240802_experiment_templates.csv'
            
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
                df = pd.concat([df, new_template], ignore_index=True)
            else:
                df = new_template
            
            df.to_csv(save_path, index=False)
            st.success(f'Template saved successfully! Start Seq Num: {start_seq_num}')

    st.markdown("""
        <div class='section'>
            <h2>Saved Templates</h2>
        </div>
    """, unsafe_allow_html=True)

    save_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/admin/20240802_experiment_templates.csv'
    if os.path.exists(save_path):
        templates_df = pd.read_csv(save_path)
        # Move description column to the first position
        columns = ['Description'] + [col for col in templates_df.columns if col != 'Description']
        templates_df = templates_df[columns]
        templates_df = st.data_editor(templates_df, key='templates_editor', num_rows="dynamic")
        templates_df.to_csv(save_path, index=False)
    else:
        st.info("No templates found. Create your first template above.")

display_admin_tab()