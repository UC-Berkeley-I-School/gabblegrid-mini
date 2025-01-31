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
from typing import Dict, List, Any, Tuple
import random
import colorsys
import base64
from PIL import Image
import io

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


def generate_contrasting_colors(num_colors):
    colors = []
    hue_step = 1.0 / num_colors
    for i in range(num_colors):
        hue = i * hue_step
        saturation = 0.4  # Lower saturation for pastel colors
        lightness = 0.8  # Higher lightness for pastel colors
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors

def create_running_girl_base64(height: int = 40):
    image_path = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/11.Progress_Bar/20240828_running_girl.png"
    with Image.open(image_path) as img:
        aspect_ratio = img.width / img.height
        new_width = int(height * aspect_ratio)
        img = img.resize((new_width, height))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

def create_stacked_progress_bar(team_progress: dict, team_colors: dict, runner_height: int = 40) -> str:
    total_width = 100
    bars = []
    overall_progress = sum(team_progress.values()) / len(team_progress)
    runner_position = overall_progress * total_width

    for team, progress in team_progress.items():
        width = progress * (total_width / len(team_progress))
        bars.append(f'<div style="width:{width}%;height:20px;background-color:{team_colors[team]};float:left;"></div>')

    running_girl_base64 = create_running_girl_base64(runner_height)
    
    return f'''
    <div style="width:100%;background-color:#eee;position:relative;height:20px;">
        {"".join(bars)}
        <div style="position:relative;height:0px;">
            <img src="data:image/png;base64,{running_girl_base64}" 
                 style="position:absolute;left:{runner_position}%;bottom:0;
                        transform:translateX(-50%);height:{runner_height}px;" />
        </div>
    </div>
    '''

def create_combined_progress_bar(selected_teams: list, runner_height: int = 40) -> dict:
    progress_container = st.empty()
    log_container = st.empty()
    team_colors = dict(zip(selected_teams, generate_contrasting_colors(len(selected_teams))))
    return {
        "progress": progress_container,
        "log": log_container,
        "team_progress": {team: 0 for team in selected_teams},
        "current_step": None,
        "team_colors": team_colors,
        "runner_height": runner_height
    }

async def update_combined_progress(team: str, step: str, progress_data: dict, team_steps: dict):
    steps = team_steps[team]
    step_index = steps.index(step)

    # Update progress to show the current step as in progress
    progress_data["team_progress"][team] = step_index / len(steps)
    progress_data["current_step"] = (team, step)

    # Update progress bar
    progress_bar_html = create_stacked_progress_bar(
        progress_data["team_progress"], 
        progress_data["team_colors"], 
        progress_data["runner_height"]
    )
    progress_data["progress"].markdown(progress_bar_html, unsafe_allow_html=True)


    # Update log with in-progress icon
    log_html = create_step_log(progress_data["team_progress"], team_steps, progress_data["current_step"])
    progress_data["log"].markdown(log_html, unsafe_allow_html=True)

    # Simulate step completion
    await asyncio.sleep(0.5)  # Add a small delay for visual effect

    # Update progress to show the step as completed
    progress_data["team_progress"][team] = (step_index + 1) / len(steps)
    progress_data["current_step"] = None

    # Update log with completed icon
    log_html = create_step_log(progress_data["team_progress"], team_steps, progress_data["current_step"])
    progress_data["log"].markdown(log_html, unsafe_allow_html=True)

    # If this is the last step, ensure the progress bar reaches 100%
    if step_index == len(steps) - 1:
        progress_data["team_progress"][team] = 1.0
        progress_bar_html = create_stacked_progress_bar(progress_data["team_progress"], progress_data["team_colors"])
        progress_data["progress"].markdown(progress_bar_html, unsafe_allow_html=True)



def create_step_log(team_progress: Dict[str, float], team_steps: Dict[str, List[str]], current_step: Tuple[str, str] = None) -> str:
    log_entries = []
    for team, progress in team_progress.items():
        completed_steps = int(progress * len(team_steps[team]))
        for i, step in enumerate(team_steps[team]):
            if i < completed_steps:
                status = "✅"  # Checkmark for completed steps
            elif current_step and current_step == (team, step):
                status = "🔄"  # In progress for current step
            else:
                status = "⬜"  # Empty square for future steps
            log_entries.append(f"<li>{team} - {status} {step}</li>")
    return f'<ul style="height:200px;overflow-y:scroll;">{"".join(log_entries)}</ul>'

######################################################################################################

def display_playground_tab():
    # Initialize session state variables
    if 'run_inference' not in st.session_state:
        st.session_state['run_inference'] = False
    if 'button_state' not in st.session_state:
        st.session_state['button_state'] = 'Run Inference'  # Initial button state
    if 'button_color' not in st.session_state:
        st.session_state['button_color'] = 'lightgreen'  # Initial button color
    if 'seen_messages' not in st.session_state:
        st.session_state['seen_messages'] = set()
    if 'show_chat_content' not in st.session_state:
        st.session_state['show_chat_content'] = False

    # Function to reset the inference and variables but stay on the same tab
    def reset_inference():
        st.session_state['run_inference'] = False
        st.session_state['button_state'] = 'Run Inference'
        st.session_state['button_color'] = 'lightgreen'
        st.session_state['inference_done'] = False
        st.session_state['template_selected'] = "No template"  # Clear the selected template
        st.session_state['parameters_disabled'] = False  # Make all parameters editable


    # Function to handle the inference process and change the button state to reset
    def run_inference():
        st.session_state['run_inference'] = True
        st.session_state['button_state'] = 'Reset'
        st.session_state['button_color'] = '#F4D03F'  # Pastel yellow/chrome color
        st.session_state['inference_done'] = True

    min_time, max_time = get_time_range(input_file)   

    # Create a centralized experiment ID
    experiment_id = create_experiment_id()

    # Custom CSS for the button styling based on state
    st.markdown(f"""
        <style>
            .stButton button {{
                font-size: 18px;
                font-weight: bold;
                color: black !important;
                background-color: {st.session_state['button_color']} !important;
                padding: 10px 24px;
                width: 180px !important;  /* Fixed width for both buttons */
                height: 48px !important;  /* Fixed height for both buttons */
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition-duration: 0.4s;
                transition: transform 0.3s ease;
            }}
            .stButton button:hover {{
                background-color: green !important; /* Dark green on hover for both states */
                color: white !important;
                transform: scale(1.1);
            }}
            .stButton button:active {{
                background-color: {st.session_state['button_color']} !important; /* Same as current state */
                color: black !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    model_config = render_sidebar('playground')

    # Render the intro section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(playground_intro, unsafe_allow_html=True)

    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/05.Playground/playground_1.gif', use_column_width=True)

    st.markdown("<style> div.block-container { padding-top: 0px; } </style>", unsafe_allow_html=True)

    # Add a header for the How-To section
    st.markdown("<h4 style='color: grey;'>How-To</h4>", unsafe_allow_html=True)
    st.markdown("<p>This section provides a collection of tutorial videos to guide you through the playground features.</p>", unsafe_allow_html=True)

    # Add the video gallery in an expander
    with st.expander("Video / Tutorial Gallery"):
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("<h5 style='color: grey;'>Playground 1</h5>", unsafe_allow_html=True)
            st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/playground_01.mp4')
    
            st.markdown("<h5 style='color: grey;'>Playground 3</h5>", unsafe_allow_html=True)
            st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/playground_03.mp4')
    
        with col2:
    
            st.markdown("<h5 style='color: grey;'>Playground 2</h5>", unsafe_allow_html=True)
            st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/playground_02.mp4')



    
    # Begin form to avoid re-executing app unnecessarily
    with st.form(key='inference_form'):
        # Render an h4 header for Agent Teams Selection
        st.markdown("<h4 style='color: grey;'>Agent Teams Selection</h4>", unsafe_allow_html=True)

        # Render Agent Teams Selection within the form
        with st.expander("Select Teams with specific goals", expanded=True):
            team_selection = render_agent_team_selection()

        # Render an h4 header for Parameters Selection
        st.markdown("<h4 style='color: grey;'>Parameters Selection</h4>", unsafe_allow_html=True)

        # Render Parameters within the form
        inference_params = render_main_content(min_time, max_time)

        # "Run Inference" / "Reset" button within the form based on state
        if st.session_state['button_state'] == 'Run Inference':
            submitted = st.form_submit_button("Run Inference", on_click=run_inference)
        else:
            reset = st.form_submit_button("Reset", on_click=reset_inference)

    # Handle the actual inference process
    if inference_params and st.session_state.run_inference:
        team_steps = get_team_steps()
        progress_data = create_combined_progress_bar(team_selection, runner_height=40)

        async def run_all():
            try:
                openai_api_key = st.session_state.get("openai_api_key", "")
                if not openai_api_key:
                    st.error("OpenAI API key is not set. Please set it in the session state.")

                # Dynamically create tabs based on selected teams
                tabs = st.tabs(team_selection)

                # Create a dictionary to map tab names to tab objects
                tab_dict = dict(zip(team_selection, tabs))

                for team in team_selection:
                    with tab_dict[team]:
                        if team == "Team 1: Basic Log Inference":
                            for step in team_steps[team]:
                                await update_combined_progress(team, step, progress_data, team_steps)
                                if step == "Running log inference":
                                    log_results = await run_log_inference(model_config, inference_params, openai_api_key)
                                elif step == "Displaying log results":
                                    if log_results and 'error' in log_results:
                                        st.error(f"Error in log inference: {log_results['error']}")
                                    else:
                                        st.write(f"Log inference results: {log_results}")

                        elif team == "Team 2: Current Weather and Search":
                            for step in team_steps[team]:
                                await update_combined_progress(team, step, progress_data, team_steps)
                                if step == "Running weather inference":
                                    weather_results = await run_weather_inference(model_config, openai_api_key)
                                elif step == "Displaying weather results":
                                    if weather_results and 'error' in weather_results:
                                        st.error(f"Error in weather inference: {weather_results['error']}")
                                    else:
                                        st.write(f"Weather inference results: {weather_results}")

                        elif team == "Team 3: Historical Weather Trends":
                            for step in team_steps[team]:
                                await update_combined_progress(team, step, progress_data, team_steps)
                                if step == "Running historical weather inference":
                                    historical_weather_results = await run_historical_weather_inference(model_config, inference_params, openai_api_key, experiment_id)
                                elif step == "Displaying historical weather results":
                                    if historical_weather_results and 'error' in historical_weather_results:
                                        st.error(f"Error in historical weather inference: {historical_weather_results['error']}")
                                    else:
                                        st.write(f"Historical weather inference results: {historical_weather_results}")
                                        st.write("---> Process completed and records inserted ..")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

        asyncio.run(run_all())

        # After process completion, set button to reset state
        st.session_state.run_inference = False


