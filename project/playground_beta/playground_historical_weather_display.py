import streamlit as st
from autogen.cache import Cache
import numpy as np
import asyncio
import pandas as pd
import torch
from datetime import datetime
import autogen
from .beta_utils.autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import re
import os
from typing import Annotated

def display_errors(error_messages):
    st.markdown("<h2 style='color: tomato;'>Errors</h2>", unsafe_allow_html=True)
    st.markdown('<br>'.join(error_messages), unsafe_allow_html=True)

def display_weather_results(results):
    # st.markdown("<h2 style='color: teal;'>Historical Trends: Logs & Max Temp</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: teal;'>Team 3: Compare log inference data (logs) with historical weather trends at datacenter (Livermore, CA) </h5>", unsafe_allow_html=True)
    if isinstance(results, dict):
        if 'content' in results:
            st.markdown(results['content'])
        else:
            for key, value in results.items():
                st.markdown(f"**{key}:** {value}")
    elif isinstance(results, str):
        st.markdown(results)
    elif isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                for key, value in item.items():
                    st.markdown(f"**{key}:** {value}")
            else:
                st.write(item)
    else:
        st.write(results)

def display_agent_interactions(chat_messages):
    # st.subheader('Agents & Interaction')
    st.markdown("<h5 style='color: grey;'>Agents & Interaction</h5>", unsafe_allow_html=True)
    conversation_transcript = []

    for message in chat_messages:
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
