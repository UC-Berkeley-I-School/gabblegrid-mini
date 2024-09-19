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

    # Create two columns with a 1:1 ratio for image and text
    col1, col2 = st.columns([1, 1])

    # Left column for the image and attribution
    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/system_arch_1.gif', use_column_width=True)
        st.markdown("""
            <div class='center'>
                <a href="https://storyset.com/business">Business illustrations by Storyset</a>
            </div>
        """, unsafe_allow_html=True)

    # Right column for the text content
    with col2:
        st.markdown("""
            <div class='section'>
                <h2>Tech Stack</h2>
                <p>
                The MindMesh solution is built upon a robust technical architecture designed to ensure high availability, scalability, and resilience. 
                This architecture leverages the capabilities of autonomous AI agents to manage and enhance service reliability in cloud environments. 
                The core components of the architecture are utlined in the two sections below:
                </p>
                <p>
                The first section outlines the different foundational elements involved in the MindMesh architecture, including preprocessing tools (Apache Spark, Hadoop), modeling tools (Amazon EC2, SageMaker), and deployment components (S3, EFS, Jupyter). 
                The second image focuses on the basis for the Agent framework - i.e. Autogen. Example agents agents are shown for illustration only; the premise being that we develop teams of agents that work together to achieve an objective with interaction and management within the environment.
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
            The AutoGen framework serves as the foundation for the AI agents. These agents are designed to operate autonomously, performing tasks that enhance service reliability. Key agents include:
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