# File: agents/historical_weather/agent_initialization.py

# from .autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent
from ...utils.autogen_setup import TrackableUserProxyAgent, TrackableAssistantAgent


def initialize_historical_weather_agents(llm_config):
    user_proxy = TrackableUserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "group", "use_docker": False},
        max_consecutive_auto_reply=10,
        system_message="""Reply TERMINATE if the task has been solved to full satisfaction. Otherwise, reply CONTINUE or the reason why the task is not solved yet.""",
        llm_config=llm_config
    )

    log_historical_weather_retriever = TrackableAssistantAgent(
        name="log_historical_weather_retriever",
        system_message="""You are an agent specialized in retrieving and processing historical weather data for log analysis. Use the functions provided to fetch and analyze weather information. Reply TERMINATE when the task is done.""",
        llm_config=llm_config
    )

    log_historical_weather_writer = TrackableAssistantAgent(
        name="log_historical_weather_writer",
        system_message="""
        You are a creative writer and your job is to take the data extracted by log_historical_weather_retriever and 
        summarize the content in an email with subject header and body. Please keep the email concise and focus on the general analysis of the data that you receive.
        Specifically, since this is a log anomaly detection task, please try to analyze the data in that context.
        In producing the content, please do not use markdown headings like # or ##, and please limit the formatting to bold and italics only.
        Finally, please generate your content in response to a specific task, please generate the content just once and never more than once.
        """,
        llm_config=llm_config
    )

    log_historical_weather_plotter = TrackableAssistantAgent(
        name="log_historical_weather_plotter",
        system_message="""
        You are an agent specialized in plotting historical weather data. Use the functions provided to fetch and plot the weather information from the parquet files. 
        For plotting, please use the max temperature as the Y axis and the sample start date/time as the x axis
        with this, please draw a simple line plot that tracks the max temperature changes along the time series dimension on the x axis (ie based on sample dates)
        Reply TERMINATE when the task is done.
        """,
        llm_config=llm_config
    )

    return {
        "user_proxy": user_proxy, 
        "log_historical_weather_retriever": log_historical_weather_retriever, 
        "log_historical_weather_writer": log_historical_weather_writer, 
        "log_historical_weather_plotter": log_historical_weather_plotter
    }
