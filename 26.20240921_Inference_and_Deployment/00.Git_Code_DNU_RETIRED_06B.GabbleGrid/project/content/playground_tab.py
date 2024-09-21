import os
import streamlit as st
from pdf2image import convert_from_path
import time
import asyncio
import json
import pandas as pd
from datetime import datetime
from autogen.cache import Cache
import autogen
from utils.autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent
from utils.inference import send_log_data_and_get_model_results
from utils.plotting import create_interaction_diagram
from content.playground_text import playground_intro, key_parameters

def display_pdf_as_images(pdf_path, display_name):
    images = convert_from_path(pdf_path, first_page=0, last_page=1)
    if images:
        st.image(images[0], caption=f'{display_name} - Page 1', use_column_width=True)
        with open(pdf_path, "rb") as f:
            st.download_button(label=f"Download", data=f, file_name=os.path.basename(pdf_path))

def display_playground_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2 {
                color: grey; /* Change text color to grey */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Model Configuration")
        selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'], index=2, key='model_selectbox_playground')
        api_key = st.text_input("API Key", value="sk-proj-iQtcgUJOOf4n53Bs6uyqT3BlbkFJnEIqUeEwXjbjVMcDVqiz", type="password", key='api_key_input_playground')
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, key='temperature_slider_playground')
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=2048, key='max_tokens_input_playground')
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, key='top_p_slider_playground')

        st.header("Essential Reading")
        folder_path = "/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/documents/essential_reading"
        
        st.markdown('<div class="essential-reading">', unsafe_allow_html=True)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            if files:
                files = sorted([file for file in files if not file.startswith('.')])
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    display_name = os.path.splitext(os.path.basename(file))[0]
                    with st.expander(display_name):
                        display_pdf_as_images(file_path, display_name)
            else:
                st.markdown("No documents found.")
        else:
            st.markdown("The specified folder does not exist.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(playground_intro, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input('Start Time YYYY-MM-DD HH:MM:SS', key='start_time_input')
    with col2:
        num_tests = st.number_input('Number of Tests', min_value=1, value=10, key='num_tests_input')

    if st.button('Run Inference', key='run_inference_button'):
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        status_placeholder.text("Step 1: Initializing inference process...")
        progress_bar.progress(5)
        time.sleep(1)

        status_placeholder.text("Step 2: Displaying project information...")
        progress_bar.progress(10)
        time.sleep(1)

        status_placeholder.text("Step 3: Loading model and data...")
        progress_bar.progress(15)
        time.sleep(1)

        status_placeholder.text("Step 4: Verifying API key...")
        progress_bar.progress(20)
        if not api_key:
            st.warning('Please provide a valid OpenAI API key', icon="⚠️")
            st.stop()
        time.sleep(1)

        status_placeholder.text(f"Step 5: Configuring language model... (Selected model: {selected_model})")
        progress_bar.progress(25)
        llm_config = {
            "config_list": [
                {
                    "model": selected_model,
                    "api_key": api_key,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }
            ]
        }

        time.sleep(2)

        status_placeholder.text("Step 6: Initializing agents...")
        progress_bar.progress(30)
        user_proxy = TrackableUserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "group", "use_docker": False},
            max_consecutive_auto_reply=10,
            system_message="""Reply TERMINATE if the task been solved at full satisfaction. Otherwise, reply CONTINUE or the reason why the task is not solved yet.""",
            llm_config=llm_config
        )

        log_data_retriever = TrackableAssistantAgent(
            name="log_data_retriever",
            system_message="For all tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
            llm_config=llm_config,
        )

        log_data_writer = TrackableAssistantAgent(
            name="log_data_writer",
            system_message="""
            You are a creative writer and your job is to take the data extracted by log_data_retriever and 
            summarize the content in an email with subject header and body. Please keep the email concise and focus on the analysis of the confusion matrix.
            Specifically since this is a log anomaly detection task, please try to analyze the recall and precision of Class 1 (ie Alert).
            In producing the content, please do not use markdown headings like # or ## and so, please limit the formatting to bold and italics only.
            Finally, please generate your content in response to a specific task, please generate the content just one and never more than once.
            """,
            llm_config=llm_config,
        )
        time.sleep(1)

        status_placeholder.text("Step 7: Registering functions...")
        progress_bar.progress(35)
        user_proxy.register_for_execution()(send_log_data_and_get_model_results)
        log_data_retriever.register_for_llm(description="Run inference on test data and evaluate model performance")(send_log_data_and_get_model_results)
        time.sleep(1)

        status_placeholder.text("Step 8: Setting up group chat...")
        progress_bar.progress(40)
        groupchat = autogen.GroupChat(agents=[user_proxy, log_data_retriever, log_data_writer], messages=[], max_round=10)
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        time.sleep(1)

        if 'chat_initiated' not in st.session_state:
            st.session_state.chat_initiated = False

        if not st.session_state.chat_initiated:
            status_placeholder.text("Step 9: Preparing for inference...")
            progress_bar.progress(45)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            time.sleep(1)

            status_placeholder.text("Step 10: Running inference...")
            progress_bar.progress(50)

            async def run_chat():
                task = f"""
                Your task is to simply run the only function you have access to and return the content generated. Please use these two parameter values:
                start_time = '{start_time}'
                num_tests = {num_tests}
                In case the start_time does not exist, the function will return a list of other available times. Please select the nearest one and proceed with the task.
                Finally, in responding to a task, please generate your content just one and never more than once.
                """

                try:
                    with Cache.disk() as cache:
                        await user_proxy.a_initiate_chat(
                            manager, message=task, summary_method="reflection_with_llm", cache=cache
                        )
                except autogen.exceptions.ModelError as e:
                    st.error(f"Model produced invalid content: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

            loop.run_until_complete(run_chat())

            if not st.session_state.chat_initiated:
                loop.close()
                st.session_state.chat_initiated = False

        status_placeholder.text("Step 11: Processing results...")
        progress_bar.progress(70)
        time.sleep(1)

        results = None
        for message in user_proxy.chat_messages[manager]:
            if isinstance(message, dict) and 'content' in message:
                try:
                    content = message['content']
                    if isinstance(content, str) and content.startswith('{') and content.endswith('}'):
                        results = json.loads(content)
                        if isinstance(results, dict) and 'metrics' in results:
                            break
                except Exception as e:
                    st.error(f"Error parsing message content: {e}")
                    continue

        if results is None:
            st.error("Failed to get results from the model")
        else:
            status_placeholder.text("Step 12: Generating final report...")
            progress_bar.progress(80)
            time.sleep(1)

            status_placeholder.text("Step 13: Displaying outcome...")
            progress_bar.progress(85)
            time.sleep(1)
            # st.subheader('Agent Output - Email')
            final_event = None
            for message in reversed(user_proxy.chat_messages[manager]):
                if isinstance(message, dict) and message.get('name') == 'log_data_writer':
                    final_event = message.get('content', '')
                    break
            if final_event:
                st.markdown("<h2 style='color: teal;'>Agent Output - Email</h2>", unsafe_allow_html=True)  # Step (1) change text color to blue
            
                # Step (2) add section divider
                # st.markdown('<hr>', unsafe_allow_html=True)
            
                # New content

                # st.markdown("<h5>This is the final output produced by one of the agents in the group. For details of the process please see the section 'Agents & Interactions'.</h6>", unsafe_allow_html=True)

                st.markdown("<p style='color: tomato;'><em>This is the final output produced by one of the agents in the group. For details of the process please see the section 'Agents & Interaction'.</em></p>", unsafe_allow_html=True)

                # st.markdown("<h4>Subject: Analysis of Log Anomaly Detection Results</h4>", unsafe_allow_html=True)
                st.markdown('<hr>', unsafe_allow_html=True)
                
                st.markdown(final_event)
            else:
                st.write("No Outcome found.")

            st.markdown('<hr>', unsafe_allow_html=True)
            

            status_placeholder.text("Step 14: Displaying metrics...")
            progress_bar.progress(90)
            time.sleep(1)
            
            st.markdown("<h2 style='color: teal;'>Other Content</h3>", unsafe_allow_html=True)
            st.subheader('Metrics')
            metrics = results['metrics']
            formatted_metrics = "\n".join([
                f"Accuracy: {metrics['Accuracy']:.4f}",
                f"Precision (Class 1): {metrics['Precision (Class 1)']:.4f}",
                f"Recall (Class 1): {metrics['Recall (Class 1)']:.4f}",
                f"F1 Score: {metrics['F1 Score']:.4f}",
                f"True Positives: {metrics['True Positives']}",
                f"False Positives: {metrics['False Positives']}",
                f"True Negatives: {metrics['True Negatives']}",
                f"False Negatives: {metrics['False Negatives']}"
            ])
            st.text(formatted_metrics)

            confusion_matrix = results['confusion_matrix']
            conf_matrix_df = pd.DataFrame(
                confusion_matrix,
                index=["Actual - Normal", "Actual - Alert"],
                columns=["Predicted - Normal", "Predicted - Alert"]
            )

            st.markdown("**Confusion Matrix**")
            st.dataframe(conf_matrix_df)

            status_placeholder.text("Step 15: Displaying agent interactions...")
            progress_bar.progress(95)
            time.sleep(1)
            st.subheader('Agents & Interaction')
            conversation_transcript = []

            for message in user_proxy.chat_messages[manager]:
                if isinstance(message, dict):
                    speaker = message.get('name', 'Unknown')
                    if speaker == 'Unknown':
                        speaker = 'user_proxy'
                    content = message.get('content', '')

                    if not content and 'tool_calls' in message:
                        tool_calls = message['tool_calls']
                        for call in tool_calls:
                            if call['type'] == 'function':
                                function_name = call['function']['name']
                                arguments = call['function']['arguments']
                                content = f"***** Suggested tool call: {function_name} *****\nArguments:\n{arguments}\n****************************************************************************************************"

                    conversation_transcript.append("--------------------------------------------------------------------------------")
                    conversation_transcript.append(f"<span style='color: blue;'>{speaker} (to chat_manager):</span>\n\n{content}")

            conversation_transcript.append("--------------------------------------------------------------------------------")
            st.markdown('\n\n'.join(conversation_transcript), unsafe_allow_html=True)

            status_placeholder.text("Step 16: Displaying interaction diagrams...")
            progress_bar.progress(98)
            time.sleep(1)
            st.subheader("Interaction Map")
            interaction_diagram_path = create_interaction_diagram()
            st.image(interaction_diagram_path)

            status_placeholder.text("Step 17: Process completed successfully!")
            progress_bar.progress(100)
            time.sleep(1)

    if 'show_chat_content' not in st.session_state:
        st.session_state['show_chat_content'] = False

    if 'seen_messages' not in st.session_state:
        st.session_state['seen_messages'] = set()

    st.markdown(key_parameters, unsafe_allow_html=True)