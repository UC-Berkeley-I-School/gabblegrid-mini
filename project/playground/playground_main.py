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
from .utils.experiments import create_experiment_id
from typing import Dict, List, Any

def get_team_steps() -> Dict[str, List[str]]:
    return {
        "Team 1: Basic Log Inference": [
            "Initializing log agents",
            "Setting up log group chat",
            "Preparing log data",
            "Running log inference",
            "Processing log results",
            "Displaying log results"
        ],
        "Team 2: Current Weather and Search": [
            "Initializing weather agents",
            "Setting up weather group chat",
            "Fetching current weather data",
            "Running weather inference",
            "Processing weather results",
            "Displaying weather results"
        ],
        "Team 3: Historical Weather Trends": [
            "Initializing historical weather agents",
            "Setting up historical weather group chat",
            "Preparing historical weather data",
            "Running historical weather inference",
            "Generating weather plot",
            "Processing historical weather results",
            "Displaying historical weather results"
        ]
    }

def create_combined_progress_bar(selected_teams: List[str]) -> Dict[str, Any]:
    progress_container = st.empty()
    log_container = st.empty()
    return {
        "progress": progress_container,
        "log": log_container,
        "team_progress": {team: 0 for team in selected_teams}
    }

def update_combined_progress(team: str, step: str, progress_data: Dict[str, Any], team_steps: Dict[str, List[str]]):
    steps = team_steps[team]
    step_index = steps.index(step)
    progress_data["team_progress"][team] = (step_index + 1) / len(steps)
    
    total_progress = sum(progress_data["team_progress"].values()) / len(progress_data["team_progress"])
    
    # Update progress bar
    progress_bar_html = create_stacked_progress_bar(progress_data["team_progress"])
    progress_data["progress"].markdown(progress_bar_html, unsafe_allow_html=True)
    
    # Update log
    log_html = create_step_log(team, step, progress_data["team_progress"], team_steps)
    progress_data["log"].markdown(log_html, unsafe_allow_html=True)

def create_stacked_progress_bar(team_progress: Dict[str, float]) -> str:
    total_width = 100
    team_colors = {"Team 1: Basic Log Inference": "#FF9999", "Team 2: Current Weather and Search": "#66B2FF", "Team 3: Historical Weather Trends": "#99FF99"}
    bars = []
    for team, progress in team_progress.items():
        width = progress * (total_width / len(team_progress))
        bars.append(f'<div style="width:{width}%;height:20px;background-color:{team_colors[team]};float:left;"></div>')
    return f'<div style="width:100%;background-color:#eee;">{"".join(bars)}</div>'

def create_step_log(current_team: str, current_step: str, team_progress: Dict[str, float], team_steps: Dict[str, List[str]]) -> str:
    log_entries = []
    for team, progress in team_progress.items():
        completed_steps = int(progress * len(team_steps[team]))
        for i, step in enumerate(team_steps[team]):
            if i < completed_steps:
                status = "✅"  # Checkmark for completed steps
            elif i == completed_steps and team == current_team and step == current_step:
                status = "🔄"  # In progress for current step
            else:
                status = "⬜"  # Empty square for future steps
            log_entries.append(f"<li>{team} - {status} {step}</li>")
    return f'<ul style="height:200px;overflow-y:scroll;">{"".join(log_entries)}</ul>'

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

    # Encapsulate the team selection within the same form as the parameters
    with st.form(key='inference_form'):
        # Render Agent Teams Selection with a card-like boundary
        with st.expander("Select Teams with specific goals", expanded=True):
            team_selection = render_agent_team_selection()
    
        # Render an h4 header for Parameters
        st.markdown("<h4 style='color: grey;'>Parameters Selection</h4>", unsafe_allow_html=True)
        
        # Render Parameters (this part remains unchanged)
        inference_params = render_main_content(min_time, max_time)
        
        # Move the submit button to the form
        submitted = st.form_submit_button("Run Inference")
    
    # Handle form submission
    if submitted:
        st.session_state['run_inference'] = True

    if inference_params and st.session_state.run_inference:
        team_steps = get_team_steps()
        progress_data = create_combined_progress_bar(team_selection)

        async def run_all():
            try:
                openai_api_key = st.session_state.get("openai_api_key", "")
                if not openai_api_key:
                    st.error("OpenAI API key is not set. Please set it in the session state.")

                # Dynamically create tabs based on selected teams
                tabs = st.tabs(team_selection)  # Use full team names for tabs

                # Create a dictionary to map tab names to tab objects
                tab_dict = dict(zip(team_selection, tabs))

                for team in team_selection:
                    with tab_dict[team]:
                        if team == "Team 1: Basic Log Inference":
                            for step in team_steps[team]:
                                update_combined_progress(team, step, progress_data, team_steps)
                                if step == "Running log inference":
                                    log_results = await run_log_inference(model_config, inference_params, openai_api_key)
                                elif step == "Displaying log results":
                                    if log_results and 'error' in log_results:
                                        st.error(f"Error in log inference: {log_results['error']}")
                                    else:
                                        st.write(f"Log inference results: {log_results}")
                                await asyncio.sleep(0.5)  # Add a small delay for visual effect

                        elif team == "Team 2: Current Weather and Search":
                            for step in team_steps[team]:
                                update_combined_progress(team, step, progress_data, team_steps)
                                if step == "Running weather inference":
                                    weather_results = await run_weather_inference(model_config, openai_api_key)
                                elif step == "Displaying weather results":
                                    if weather_results and 'error' in weather_results:
                                        st.error(f"Error in weather inference: {weather_results['error']}")
                                    else:
                                        st.write(f"Weather inference results: {weather_results}")
                                await asyncio.sleep(0.5)  # Add a small delay for visual effect

                        elif team == "Team 3: Historical Weather Trends":
                            for step in team_steps[team]:
                                update_combined_progress(team, step, progress_data, team_steps)
                                if step == "Running historical weather inference":
                                    historical_weather_results = await run_historical_weather_inference(model_config, inference_params, openai_api_key, experiment_id)
                                elif step == "Displaying historical weather results":
                                    if historical_weather_results and 'error' in historical_weather_results:
                                        st.error(f"Error in historical weather inference: {historical_weather_results['error']}")
                                    else:
                                        st.write(f"Historical weather inference results: {historical_weather_results}")
                                        st.write("---> Process completed and records inserted ..")
                                await asyncio.sleep(0.5)  # Add a small delay for visual effect

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

        asyncio.run(run_all())

        # Reset run_inference after the process is complete
        st.session_state.run_inference = False