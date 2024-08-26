# playground_weather_inference.py

import streamlit as st
import asyncio
import requests
import json
from datetime import datetime
from autogen.cache import Cache
import autogen
from .utils.autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent
from autogen.agentchat.contrib.web_surfer import WebSurferAgent
from typing import Annotated
import re

def initialize_weather_agents(llm_config):
    user_proxy = TrackableUserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "group", "use_docker": False},
        max_consecutive_auto_reply=10,
        system_message="""Reply TERMINATE if the task been solved at full satisfaction. Otherwise, reply CONTINUE or the reason why the task is not solved yet.""",
        llm_config=llm_config
    )

    current_weather_data_retriever = TrackableAssistantAgent(
        name="current_weather_data_retriever",
        system_message="""You are the agent specializing in answering all weather related queries. When part of a group, please direct all such queries to yourself. Please only use the functions you have been provided with. Reply TERMINATE when the task is done.""",
        llm_config=llm_config
    )
    
    engineer = TrackableAssistantAgent(
        name="engineer",
        system_message="""For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.""",
        llm_config=llm_config,
    )

    bing_api_key = st.session_state.get("bing_api_key", "")
    if not bing_api_key:
        raise ValueError("Bing API key is missing.")
    
    web_surfer = WebSurferAgent(
        "web_surfer",
        llm_config=llm_config,
        summarizer_llm_config=llm_config,
        browser_config={"viewport_size": 4096, "bing_api_key": bing_api_key},
    )

    
    # web_surfer = WebSurferAgent(
    #     "web_surfer",
    #     llm_config=llm_config,
    #     summarizer_llm_config=llm_config,
    #     browser_config={"viewport_size": 4096, "bing_api_key": BING_API_KEY},
    # )

    return {"user_proxy": user_proxy, "current_weather_data_retriever": current_weather_data_retriever, "engineer": engineer, "web_surfer": web_surfer}

def register_weather_functions(agents):
    @agents["user_proxy"].register_for_execution()
    @agents["current_weather_data_retriever"].register_for_llm(description="function answering all queries related to the weather")
    def get_temperature_data(zipcode: str) -> dict:
        # api_key = OPENWEATHER_API_KEY
        api_key = st.session_state.get("openweather_api_key", "")
        if not api_key:
            return {"error": "OpenWeather API key is missing."}

        geocoding_url = "http://api.openweathermap.org/geo/1.0/zip"
        one_call_url = "https://api.openweathermap.org/data/3.0/onecall"
        
        geocoding_params = {"zip": zipcode, "appid": api_key}
        geo_response = requests.get(geocoding_url, params=geocoding_params)
        if geo_response.status_code != 200:
            return {"error": "Failed to fetch geographic coordinates."}
        
        geo_data = geo_response.json()
        lat, lon = geo_data.get("lat"), geo_data.get("lon")
        
        one_call_params = {
            "lat": lat,
            "lon": lon,
            "exclude": "minutely,hourly,alerts",
            "appid": api_key,
            "units": "metric"
        }
        weather_response = requests.get(one_call_url, params=one_call_params)
        if weather_response.status_code != 200:
            return {"error": "Failed to fetch weather data."}
        
        weather_data = weather_response.json()
        current = weather_data.get("current", {})
        daily = weather_data.get("daily", [])

        current_temp = current.get("temp")
        humidity = current.get("humidity")
        wind_speed = current.get("wind_speed")
        weather_description = current.get("weather", [{}])[0].get("description")
        
        daily_temps = [day.get("temp", {}).get("day") for day in daily]
        average_temp = sum(daily_temps) / len(daily_temps) if daily_temps else None

        daily_humidity = [day.get("humidity") for day in daily]
        average_humidity = sum(daily_humidity) / len(daily_humidity) if daily_humidity else None

        daily_wind_speed = [day.get("wind_speed") for day in daily]
        average_wind_speed = sum(daily_wind_speed) / len(daily_wind_speed) if daily_wind_speed else None

        return {
            "current_temperature": current_temp,
            "average_temperature": average_temp,
            "current_humidity": humidity,
            "average_humidity": average_humidity,
            "current_wind_speed": wind_speed,
            "average_wind_speed": average_wind_speed,
            "weather_description": weather_description
        }

    @agents["user_proxy"].register_for_execution()
    @agents["engineer"].register_for_llm(description="run cell in ipython and return the execution result.")
    def exec_python(cell: Annotated[str, "Valid Python cell to execute."]) -> str:
        try:
            exec_locals = {}
            exec(cell, {}, exec_locals)
            return str(exec_locals)
        except Exception as e:
            return f"Error: {e}"

    @agents["user_proxy"].register_for_execution()
    @agents["web_surfer"].register_for_llm(description="Search the web for recent news and information.")
    def search_web(query: str) -> str:
        try:
            results = agents["web_surfer"].search(query)
            summary = agents["web_surfer"].summarize_results(results)
            return summary
        except Exception as e:
            return f"Error during web search: {str(e)}"

def setup_weather_group_chat(agents, llm_config):
    groupchat = autogen.GroupChat(agents=list(agents.values()), messages=[], max_round=5)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return groupchat, manager


async def run_chat_weather_agents(user_proxy, manager):
    task = """
    1. Can you get the current weather information for Livermore, Zip Code 94550?, and plot the returned values in a table using the python tool available to you.
    2. After that, please search the web for recent news about extreme weather events in California and summarize the findings.
    Please use the appropriate tool for each task. Use the weather tool for weather-related queries and the web search tool for the news search.
    """    
# async def run_chat_weather_agents(user_proxy, manager):
#     task = "Can you get the current weather information for Livermore, Zip Code 94550?, and plot the returned values in a table using the python tool available to you. Please answer this question only using the appropriate tool when required. For example, please use the weather tool for all weather related queries. Please use the search tool only if specifically asked to search the web for an answer."

    try:
        with Cache.disk() as cache:
            await user_proxy.a_initiate_chat(
                manager, message=task, summary_method="reflection_with_llm", cache=cache
            )
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

    for message in user_proxy.chat_messages[manager]:
        if isinstance(message, dict) and 'content' in message:
            try:
                content = message['content']
                if isinstance(content, str):
                    if content.startswith('{') and content.endswith('}'):
                        results = json.loads(content)
                        if isinstance(results, dict):
                            if 'current_temperature' in results:
                                return results
                            elif 'error' in results:
                                return results
                    elif "No matching model found" in content:
                        return {"error": content}
            except Exception as e:
                st.error(f"Error parsing message content: {e}")
    
    return None

async def run_weather_inference(model_config, openai_api_key):
    error_messages = []

    if not openai_api_key:
        st.warning('Please provide a valid OpenAI API key', icon="⚠️")
        return

    # Create the llm_config with the OpenAI API key
    llm_config = {
        "config_list": [
            {
                "model": model_config["model"],
                "api_key": openai_api_key,
                "temperature": model_config["temperature"],
                "max_tokens": model_config["max_tokens"],
                "top_p": model_config["top_p"]
            }
        ]
    }
    
    try:
        agents = initialize_weather_agents(llm_config)
        # agents = initialize_weather_agents(llm_config, openai_api_key)
    except Exception as e:
        error_messages.append(f"Error initializing agents: {e}")
        return display_errors(error_messages)

    try:
        register_weather_functions(agents)
    except Exception as e:
        error_messages.append(f"Error registering functions: {e}")
        return display_errors(error_messages)

    try:
        groupchat, manager = setup_weather_group_chat(agents, llm_config)
    except Exception as e:
        error_messages.append(f"Error setting up group chat: {e}")
        return display_errors(error_messages)

    try:
        results = await run_chat_weather_agents(agents["user_proxy"], manager)
    except Exception as e:
        error_messages.append(f"Error during inference: {e}")
        return display_errors(error_messages)

    if results:
        if 'error' in results:
            return display_errors([results['error']])
        else:
            display_weather_results(results, agents["user_proxy"].chat_messages[manager])
            return results
    else:
        return display_errors(["Failed to get results from the model"])

def display_errors(error_messages):
    st.markdown("<h2 style='color: tomato;'>Errors</h2>", unsafe_allow_html=True)
    st.markdown('<br>'.join(error_messages), unsafe_allow_html=True)

def display_weather_results(results, chat_messages):
    # st.markdown("<h2 style='color: teal;'>Weather Information</h2>", unsafe_allow_html=True)
    st.markdown("<h3 class='custom-teal' style='color: teal;'>Team 2: Report on current weather conditions at datacenter (Livermore, CA) </h5>", unsafe_allow_html=True)
    st.markdown("<p style='color: cerulean;'><em>This is the final output produced by the weather agents. For details of the process please see the section 'Agents & Interaction'.</em></p>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    if results:
        st.markdown(f"**Current Temperature:** {results['current_temperature']}°C")
        st.markdown(f"**Average Temperature:** {results['average_temperature']}°C")
        st.markdown(f"**Current Humidity:** {results['current_humidity']}%")
        st.markdown(f"**Average Humidity:** {results['average_humidity']}%")
        st.markdown(f"**Current Wind Speed:** {results['current_wind_speed']} m/s")
        st.markdown(f"**Average Wind Speed:** {results['average_wind_speed']} m/s")
        st.markdown(f"**Weather Description:** {results['weather_description']}")
    else:
        st.write("No weather information found.")

    st.markdown('<hr>', unsafe_allow_html=True)
    
#######################################################################################    
    
    st.markdown("<p style='font-size:22px; color:grey;'>Web Search Results</p>", unsafe_allow_html=True)
    web_search_found = False
    for message in chat_messages:
        if isinstance(message, dict) and 'name' in message and 'content':
            if message['name'] == 'web_surfer':
                content = message['content']
                # Remove or replace headers
                content = re.sub(r'(Web Results|News Results|=======================)', '', content)
                
                # Format the content while preserving links
                formatted_content = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        # Preserve links
                        link_match = re.match(r'(\d+\.\s*)?(\[.*?\]\((.*?)\))(.*)', line)
                        if link_match:
                            number, link_text, url, rest = link_match.groups()
                            # Set hyperlink color to blue and underline
                            formatted_line = f"{number or ''}<a href='{url}' style='color: blue; text-decoration: underline;'>{link_text[1:-1]}</a>{rest}"
                        else:
                            formatted_line = line
                        # Increase the font size for the content
                        formatted_content.append(f"<p style='font-size:16px; color:grey; margin: 0;'>{formatted_line}</p>")
                
                st.markdown("\n".join(formatted_content), unsafe_allow_html=True)
                web_search_found = True

    if not web_search_found:
        st.markdown("No web search results found.")
    
    display_agent_interactions(chat_messages)

#############################################################################################

def display_agent_interactions(chat_messages):
    st.markdown("<h4 style='color: grey;'>Agents & Interaction</h4>", unsafe_allow_html=True)
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
            # Process content to style headers differently
            processed_content = re.sub(
                r'^(.*Results:?)$',  # Matches lines that end with "Results" or "Results:"
                r'<span style="font-size:18px; font-weight:bold;">\1</span>',
                content,
                flags=re.MULTILINE
            )
            # Apply a consistent style to all interaction text
            conversation_transcript.append(
                f"<div style='font-size:16px; color: grey; margin-bottom:14px; font-weight: normal;'>"
                f"<span style='color: blue; font-weight:normal;'>{speaker} (to chat_manager):</span><br>"
                f"{processed_content}"
                f"</div>"
            )
    st.markdown('\n\n'.join(conversation_transcript), unsafe_allow_html=True)