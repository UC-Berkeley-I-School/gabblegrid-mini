import streamlit as st
import asyncio
from utils.sidebar_utils import csv_path, input_file

from utils.sidebar_utils import render_sidebar
from .playground_ui import render_main_content, render_agent_team_selection
from .playground_log_inference import run_log_inference
from .playground_weather_inference import run_weather_inference
from playground.utils.parameter_sourcing import get_time_range
from .playground_text import playground_intro, playground_intro_expanded
from .playground_historical_weather_main import run_historical_weather_inference
from .utils.experiments import create_experiment_id  # Import the new experiment ID function


async def update_progress(progress_bar, status_placeholder, stop_event, steps):
    for i, step in enumerate(steps):
        if stop_event.is_set():
            break
        status_placeholder.text(f"Step {i+1}: {step}")
        progress_bar.progress((i+1) / len(steps))
        await asyncio.sleep(0.5)

def display_playground_tab():
    # Initialize session state variables
    if 'run_inference' not in st.session_state:
        st.session_state['run_inference'] = False
    if 'seen_messages' not in st.session_state:
        st.session_state['seen_messages'] = set()
    if 'show_chat_content' not in st.session_state:
        st.session_state['show_chat_content'] = False

    min_time, max_time = get_time_range(input_file)   

    # Create a centralized experiment ID
    experiment_id = create_experiment_id()
    
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .hardcoded-param p {
                color: grey;
            }
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    model_config = render_sidebar('playground')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(playground_intro, unsafe_allow_html=True)
    
    with col2:
        st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/20240805_Final.mp4')

    with st.expander("Read more"):
        st.markdown(playground_intro_expanded, unsafe_allow_html=True)

    # Render an h4 header for Agent Teams Selection
    st.markdown("<h4 style='color: grey;'>Agent Teams Selection</h4>", unsafe_allow_html=True)

    # Render Agent Teams Selection with a card-like boundary
    with st.expander("Select Teams with specific goals", expanded=True):
        team_selection = render_agent_team_selection()

    # Render an h4 header for Parameters
    st.markdown("<h4 style='color: grey;'>Parameters Selection</h4>", unsafe_allow_html=True)

    # Encapsulate all parameter selection within a form
    with st.form(key='inference_form'):
        inference_params = render_main_content(min_time, max_time)
        submitted = st.form_submit_button("Run Inference")
    
    if submitted:
        st.session_state['run_inference'] = True

    if inference_params and st.session_state.run_inference:
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        stop_event = asyncio.Event()

        async def run_all():
            steps = [
                "Initializing inference process...",
                "Displaying project information...",
                "Loading model and data...",
                "Verifying API key...",
                "Configuring language model...",
                "Initializing agents...",
                "Registering functions...",
                "Setting up group chat...",
                "Preparing for log inference...",
                "Running log inference...",
                "Processing log inference results...",
                "Displaying log inference results...",
                "Preparing for weather inference...",
                "Running weather inference...",
                "Processing weather inference results...",
                "Displaying weather inference results...",
                "Process completed successfully!"
            ]
            
            progress_task = asyncio.create_task(update_progress(progress_bar, status_placeholder, stop_event, steps))
            try:
                
                ###################### Import API Keys ########################################
                openai_api_key = st.session_state.get("openai_api_key", "")  # Ensure the key is retrieved and assigned
                if not openai_api_key:
                    st.error("OpenAI API key is not set. Please set it in the session state.")

                # Dynamically create tabs based on selected teams
                tabs = st.tabs(team_selection)  # Use full team names for tabs

                # Create a dictionary to map tab names to tab objects
                tab_dict = dict(zip(team_selection, tabs))

                # Basic Log Inference
                if "Team 1: Basic Log Inference" in team_selection:
                    with tab_dict['Team 1: Basic Log Inference']:
                        log_results = await run_log_inference(model_config, inference_params, openai_api_key)
                        st.write(f"Log inference results: {log_results}")  # Debug: log the inference results

                        if log_results and 'error' in log_results:
                            stop_event.set()
                            status_placeholder.text(f"Error occurred: {log_results['error']}")

                # Current Weather and Search
                if "Team 2: Current Weather and Search" in team_selection:
                    with tab_dict['Team 2: Current Weather and Search']:
                        weather_results = await run_weather_inference(model_config, openai_api_key)
                        st.write(f"Weather inference results: {weather_results}")  # Debug: log the inference results

                        if weather_results and 'error' in weather_results:
                            stop_event.set()
                            status_placeholder.text(f"Error occurred: {weather_results['error']}")
                            st.error(f"Error: {weather_results['error']}")

                # Historical Weather Trends
                if "Team 3: Historical Weather Trends" in team_selection:
                    with tab_dict['Team 3: Historical Weather Trends']:
                        historical_weather_results = await run_historical_weather_inference(model_config, inference_params, openai_api_key, experiment_id)
                        st.write(f"Historical weather inference results: {historical_weather_results}")  # Debug: log the inference results

                        if historical_weather_results and 'error' in historical_weather_results:
                            stop_event.set()
                            status_placeholder.text(f"Error occurred: {historical_weather_results['error']}")
                            st.error(f"Error: {historical_weather_results['error']}")
                        else:
                            st.write("---> Process completed and records inserted ..")
            
            except Exception as e:
                stop_event.set()
                status_placeholder.text(f"An unexpected error occurred: {str(e)}")
                st.error(f"An unexpected error occurred: {str(e)}")
            finally:
                stop_event.set()
            await progress_task
        
        asyncio.run(run_all())

        # Reset run_inference after the process is complete
        st.session_state.run_inference = False