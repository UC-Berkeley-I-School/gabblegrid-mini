import streamlit as st

def display_product_vision_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section table, .section td, .section th {
                color: grey; /* Change text color to grey */
            }
            details summary {
                color: grey; /* Change 'read more' text color to grey */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # New Section 1 - Vision
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/product_vision_1.gif', use_column_width=True)
        st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='section'>
                <h2>Vision</h2>
                <p>
                    GabbleGrid aims to reduce the impact of service outages by deploying autonomous AI agents to enhance service resiliency. Our system proactively detects and resolves issues, ensuring continuous operations.
                </p>
                <p>
                    These AI agents autonomously monitor logs, analyze data, and make real-time decisions. By detecting anomalies and taking proactive actions, they maintain service availability while reducing the need for human intervention.
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 2 - System Components Summary with Image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
            <div class='section'>
                <h2>Components</h2>
                <p>
                    The key components of our Agent system include:
                    <ul>
                        <li>Language Models</li>
                        <li>Autonomous Agents</li>
                        <li>Anomaly Detection Models</li>
                        <li>Data Sources</li>
                        <li>Server/Pod</li>
                    </ul>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/product_vision_2.gif', use_column_width=True)
        st.markdown("""
            <div class='center'>
                <a href="https://storyset.com/business">Business illustrations by Storyset</a>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################

    # Section 3 - Detailed System Components
    st.markdown("""
        <div class='section'>
            <h2>Explained</h2>
            <p>
                The Agent system integrates multiple components to manage and optimize operations:
                <ul>
                    <li><strong>Language Models:</strong> Enable the agents to understand and process complex instructions and data.</li>
                    <li><strong>Autonomous Agents:</strong> Operate independently to manage tasks and make decisions based on real-time data.</li>
                    <li><strong>Anomaly Detection Models:</strong> Identify and respond to irregularities and potential issues in the system.</li>
                    <li><strong>Data Sources:</strong> 
                        <ul>
                            <li><strong>Hardware Data:</strong> Includes metrics like CPU usage, memory, and storage I/O (SLI-1 and SLI-2).</li>
                            <li><strong>Software Data:</strong> Comprises event logs and security logs (SLI-3).</li>
                            <li><strong>External Data:</strong> Integrates external factors such as weather data.</li>
                        </ul>
                    </li>
                    <li><strong>Server/Pod:</strong> The central unit where all data is processed and decisions are executed.</li>
                </ul>
                This integrated system enables proactive management and optimization, ensuring high performance and reliability in both experimental and production environments.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_20.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)


#########################################################################################################

    # Section 4 - Demo Scope (Playground)
    st.markdown("""
        <div class='section'>
            <h2>Demo Scope</h2>
            <p>
                The demo environment, referred to as the Playground, allows users to interact with the Agent system and experience its capabilities in real time. 
                This demo showcases three different teams of agents, each performing various tasks from simple inference to more complex actions like synthesizing historical data.
            </p>
            <p>
                The Playground scope includes:
                <ol>
                    <li><strong>Simple Inference:</strong> The agent team evaluates event logs, detects anomalies, and makes real-time decisions on potential issues, as visualized in the diagram.</li>
                    <li><strong>Email Composition:</strong> Another agent team automatically drafts alert emails based on detected anomalies and system events, enabling quick response actions.</li>
                    <li><strong>Weather Data Integration:</strong> Agents can retrieve current weather information and also synthesize historical weather data, allowing for context-aware decision-making.</li>
                </ol>
                This demo provides a comprehensive hands-on experience, allowing users to try out the concept in a controlled environment where agents operate autonomously and collaboratively across multiple domains.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Image Display
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Agents/Agents_21.png', use_column_width=True)
    
    # End with a horizontal rule
    st.markdown("<hr>", unsafe_allow_html=True)
