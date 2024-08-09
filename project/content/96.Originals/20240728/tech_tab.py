import streamlit as st

def display_tech_tab():
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

    st.markdown("""
        <div class='section'>
            <h2>Technical Architecture</h2>
            <p>
            The GabbleGrid solution is built upon a robust technical architecture designed to ensure high availability, scalability, and resilience. This architecture leverages the capabilities of autonomous AI agents to manage and enhance service reliability in cloud environments. The core components of our architecture are:
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>System Components</h2>
            <p>
            The core infrastructure is meticulously crafted on EC2 instances, providing the necessary computational power and flexibility. Below are the key system components:
            </p>
            <ul>
                <li><strong>Data Ingestion:</strong> Collects and parses system logs, ensuring real-time data availability.</li>
                <li><strong>Feature Engineering:</strong> Processes raw log data to create time-based and synthetic features essential for anomaly detection.</li>
                <li><strong>Model Training:</strong> Utilizes advanced machine learning models, including Transformers, to train on historical data and predict future anomalies.</li>
                <li><strong>Inference Engine:</strong> Deploys trained models to analyze incoming log data and detect anomalies in real-time.</li>
                <li><strong>Agent Framework:</strong> The AutoGen-based framework for autonomous agents that perform specific tasks, such as data retrieval, model inference, and communication.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Architecture_Overall.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Agent Architecture</h2>
            <p>
            The AutoGen framework serves as the foundation for our AI agents. These agents are designed to operate autonomously, performing tasks that enhance service reliability. Key agents include:
            </p>
            <ul>
                <li><strong>User Proxy Agent:</strong> Interfaces with users to gather inputs and provide outputs, ensuring a seamless interaction experience.</li>
                <li><strong>Group Manager Agent:</strong> Coordinates the activities of multiple agents, ensuring efficient task execution.</li>
                <li><strong>Model Inference Agent:</strong> Executes model inference tasks to detect anomalies in real-time log data.</li>
                <li><strong>Communication Agent:</strong> Generates alerts and notifications based on model inference results, ensuring timely communication with relevant stakeholders.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Architecture_Agents.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Future Enhancements</h2>
            <p>
            Our vision for GabbleGrid includes continuous improvements to our technical architecture. Future enhancements will focus on expanding the capabilities of our AI agents, integrating more advanced machine learning models, and refining our data processing pipelines to handle even larger and more complex datasets.
            </p>
        </div>
    """, unsafe_allow_html=True)