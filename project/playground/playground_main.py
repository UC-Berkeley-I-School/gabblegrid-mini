import streamlit as st
import asyncio
from utils.sidebar_utils import csv_path, input_file

from utils.sidebar_utils import render_sidebar#, render_essential_reading

from .playground_ui import render_main_content

from .playground_inference import run_inference
from utils.parameter_sourcing import get_time_range
from .playground_text import playground_intro, playground_intro_expanded, key_parameters  # Updated import

async def update_progress(progress_bar, status_placeholder, stop_event):
    steps = [
        "Initializing inference process...",
        "Displaying project information...",
        "Loading model and data...",
        "Verifying API key...",
        "Configuring language model...",
        "Initializing agents...",
        "Registering functions...",
        "Setting up group chat...",
        "Preparing for inference...",
        "Running inference...",
        "Processing results...",
        "Generating final report...",
        "Displaying outcome...",
        "Displaying metrics...",
        "Displaying agent interactions...",
        "Displaying interaction diagrams...",
        "Process completed successfully!"
    ]
    for i, step in enumerate(steps):
        if stop_event.is_set():
            break
        status_placeholder.text(f"Step {i+1}: {step}")
        progress_bar.progress((i+1) / len(steps))
        await asyncio.sleep(0.5)  # Simulate some work being done

def display_playground_tab():
    if 'seen_messages' not in st.session_state:
        st.session_state.seen_messages = set()
    if 'chat_initiated' not in st.session_state:
        st.session_state.chat_initiated = False
    if 'show_chat_content' not in st.session_state:
        st.session_state.show_chat_content = False

    min_time, max_time = get_time_range(input_file)
    
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
    # model_config = render_sidebar()

    
    st.markdown(playground_intro, unsafe_allow_html=True)
    
    with st.expander("Read more"):
        st.markdown(playground_intro_expanded, unsafe_allow_html=True)
    
    inference_params = render_main_content(min_time, max_time)

    # Access the API key from session_state
    api_key = st.session_state.get("api_key", "")
    st.write("API Key accessed:", api_key)  # Debug print


    
    if st.button('Run Inference', key='run_inference_button'):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        stop_event = asyncio.Event()
    
        async def run_all():
            progress_task = asyncio.create_task(update_progress(progress_bar, status_placeholder, stop_event))
            try:
                # results = await run_inference(model_config, inference_params)
                # Update the call to run_inference to include api_key
                results = await run_inference(model_config, inference_params, api_key=api_key)
                if results:
                    if 'error' in results:
                        stop_event.set()
                        status_placeholder.text(f"Error occurred: {results['error']}")
                        st.error(f"Error: {results['error']}")
                    # No need for an else clause here, as display_results is called within run_inference
                else:
                    stop_event.set()
                    status_placeholder.text("Failed to get results from the model")
                    st.error("Failed to get results from the model")
            except Exception as e:
                stop_event.set()
                status_placeholder.text(f"An unexpected error occurred: {str(e)}")
                st.error(f"An unexpected error occurred: {str(e)}")
            finally:
                stop_event.set()
            await progress_task
        
        asyncio.run(run_all())