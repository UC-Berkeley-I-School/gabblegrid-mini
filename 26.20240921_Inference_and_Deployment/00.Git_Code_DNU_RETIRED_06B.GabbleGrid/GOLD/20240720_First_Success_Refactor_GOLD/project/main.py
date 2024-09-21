import streamlit as st
import time
import asyncio
import autogen
import json
from datetime import datetime
from model.load_model import load_model
from utils.autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent
from utils.inference import send_log_data_and_get_model_results
from utils.plotting import create_interaction_diagram
from autogen.cache import Cache  # Add this import

def main():
    st.set_page_config(page_title="GabbleGrid", page_icon="🔍", layout="wide")

    # New static section at the top
    st.markdown("""
        <h2 style='color: #4169E1; font-size: 28px;'>GabbleGrid: Proactive Cloud Service Reliability</h2>
        <p style='font-size: 16px;'>
        Developed as a UC Berkeley graduate capstone, GabbleGrid revolutionizes cloud service management. Our solution employs autonomous agents to:
        <ol>
            <li>Analyze log data</li>
            <li>Select optimal ML models</li>
            <li>Execute preemptive actions</li>
        </ol>
        By predicting and preventing failures, GabbleGrid enhances uptime, reliability, and user experience. We're seeking cloud service provider partnerships to validate our solution in real-world environments.
        </p>
        """, unsafe_allow_html=True)

    # Parameter input boxes
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input('Start Time YYYY-MM-DD HH:MM:SS', key='start_time_input')
    with col2:
        num_tests = st.number_input('Number of Tests', min_value=1, value=10, key='num_tests_input')

    if 'show_chat_content' not in st.session_state:
        st.session_state['show_chat_content'] = False

    if 'seen_messages' not in st.session_state:
        st.session_state['seen_messages'] = set()

    with st.sidebar:
        st.header("OpenAI Configuration")
        selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'], index=1)
        api_key = "sk-proj-iQtcgUJOOf4n53Bs6uyqT3BlbkFJnEIqUeEwXjbjVMcDVqiz"

    if st.button('Run Inference'):
        # Create placeholders for status updates
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        # Step 1: Initialize inference process
        status_placeholder.text("Step 1: Initializing inference process...")
        progress_bar.progress(5)
        time.sleep(1)

        # Step 2: Display project information
        status_placeholder.text("Step 2: Displaying project information...")
        progress_bar.progress(10)
        time.sleep(1)

        # Step 3: Load model and data
        status_placeholder.text("Step 3: Loading model and data...")
        progress_bar.progress(15)
        time.sleep(1)
        st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/10.Image_and_Static_Content/IMG_0252.jpeg')

        # Step 4: Verify API key
        status_placeholder.text("Step 4: Verifying API key...")
        progress_bar.progress(20)
        if not api_key:
            st.warning('Please provide a valid OpenAI API key', icon="⚠️")
            st.stop()
        time.sleep(1)

        # Step 5: Configure LLM
        status_placeholder.text("Step 5: Configuring language model...")
        progress_bar.progress(25)
        llm_config = {
            "config_list": [
                {
                    "model": selected_model,
                    "api_key": api_key
                }
            ]
        }
        time.sleep(2)

        # Step 6: Initialize agents
        status_placeholder.text("Step 6: Initializing agents...")
        progress_bar.progress(30)
        user_proxy = TrackableUserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "group",
                "use_docker": False,
            },
            max_consecutive_auto_reply=10,
            system_message="""Reply TERMINATE if the task been solved at full satisfaction. 
            Otherwise, reply CONTINUE or the reason why the task is not solved yet. """,
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
            summarize the content in a email with subject header and body. Please keep the email concise and focus on the analysis of the confusion matrix.
            Specifically since this is a log anomaly detection task, please try to analyze the recall and precision of Class 1 (ie Alert)
            """,
            llm_config=llm_config,
        )
        time.sleep(1)

        # Step 7: Register functions
        status_placeholder.text("Step 7: Registering functions...")
        progress_bar.progress(35)
        user_proxy.register_for_execution()(send_log_data_and_get_model_results)
        log_data_retriever.register_for_llm(description="Run inference on test data and evaluate model performance")(send_log_data_and_get_model_results)
        time.sleep(1)

        # Step 8: Set up group chat
        status_placeholder.text("Step 8: Setting up group chat...")
        progress_bar.progress(40)
        groupchat = autogen.GroupChat(agents=[user_proxy, log_data_retriever, log_data_writer], messages=[], max_round=10)
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        time.sleep(1)

        if 'chat_initiated' not in st.session_state:
            st.session_state.chat_initiated = False

        if not st.session_state.chat_initiated:
            # Step 9: Prepare for inference
            status_placeholder.text("Step 9: Preparing for inference...")
            progress_bar.progress(45)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            time.sleep(1)

            # Step 10: Run inference
            status_placeholder.text("Step 10: Running inference...")
            progress_bar.progress(50)

            async def run_chat():
                task = f"""
                Your task is to simply run the only function you have access to and return the content generated. Please use these two parameter values:
                start_time = '{start_time}'
                num_tests = {num_tests}
                In case the start_time does not exist, the function will return a list of other available times. Please select the nearest one and proceed with the task.
                """
                
                with Cache.disk() as cache:
                    await user_proxy.a_initiate_chat(
                        manager, message=task, summary_method="reflection_with_llm", cache=cache
                    )

            loop.run_until_complete(run_chat())
            loop.close()

            st.session_state.chat_initiated = True

        # Step 11: Process results
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
            # Step 12: Generate final report
            status_placeholder.text("Step 12: Generating final report...")
            progress_bar.progress(80)
            time.sleep(1)

            # Step 13: Display outcome
            status_placeholder.text("Step 13: Displaying outcome...")
            progress_bar.progress(85)
            time.sleep(1)
            st.subheader('Outcome')
            final_event = None
            for message in reversed(user_proxy.chat_messages[manager]):
                if isinstance(message, dict) and message.get('name') == 'log_data_writer':
                    final_event = message.get('content', '')
                    break
            if final_event:
                st.markdown(final_event)
            else:
                st.write("No Outcome found.")

            # Step 14: Display metrics
            status_placeholder.text("Step 14: Displaying metrics...")
            progress_bar.progress(90)
            time.sleep(1)
            st.subheader('Metrics')
            st.json(results['metrics'])
            
            st.subheader('Confusion Matrix')
            st.table(results['confusion_matrix'])
            
            # Step 15: Display agent interactions
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
            
            # Step 16: Display interaction diagrams
            status_placeholder.text("Step 16: Displaying interaction diagrams...")
            progress_bar.progress(98)
            time.sleep(1)
            st.subheader("Diagram 1: Initial Interaction Diagram")
            st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/10.Image_and_Static_Content/IMG_0252.jpeg')
            
            st.subheader("Diagram 2: Interaction Steps Diagram")
            interaction_diagram_path = create_interaction_diagram()
            st.image(interaction_diagram_path)

            # Step 17: Process completion
            status_placeholder.text("Step 17: Process completed successfully!")
            progress_bar.progress(100)
            time.sleep(1)

if __name__ == '__main__':
    main()
