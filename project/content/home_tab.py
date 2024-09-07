import streamlit as st

def display_home_tab():
    # Add the news headline image collage at the top
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/11.Site_Content_Files/11.Images/01.Banner/Logo1.png', caption='', use_column_width=True)
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section h5, .section table, .section td, .section th {
                color: grey; /* Change text color to grey */
            }
            .grey-text {
                color: grey; /* Explicit class for grey text */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
            /* Styles for MVP Concept subsections */
            .mvp-subsection {
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <hr>
        <div class='section'>
            <p>
                Developed as a UC Berkeley graduate capstone initiative, mindmesh.io transforms cloud service management with autonomous AI agents that:
                <ol>
                    <li>Analyze vast amounts of log data in real-time</li>
                    <li>Select and optimize machine learning models for precise anomaly detection</li>
                    <li>Execute proactive measures to prevent service disruptions</li>
                </ol>
            </p>
            <p class='grey-text'>Mindmesh addresses the challenges of service outages in complex IT infrastructures by leveraging the latest advancements in AI and machine learning. Our autonomous agents perform real-time monitoring, predictive maintenance, and proactive anomaly detection, significantly reducing the frequency and impact of service disruptions, thereby ensuring continuous and reliable operations.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='section'>
            <h2>The Cost of Service Outages</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/11.Site_Content_Files/11.Images/03.gifs/business_case_1.gif', use_column_width=True)
    st.markdown("""
    <div class='center'>
        <a href="https://storyset.com/business">Business illustrations by Storyset</a>
    </div>
""", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
            <p class='grey-text'>
            In a cloud-native world, containerized applications and microservices are the norm. While they offer benefits like flexibility and scalability, they're also far more dynamic and complex. Unplanned downtime is significant and pervasive, affecting 80% of organizations within the last three years, with 76% enduring downtime that led to data loss in 2021 alone. The costs associated with these outages include lost productivity, reputational damage, and direct revenue loss.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Challenges and Solution section
    st.markdown("""
        <hr>
        <div class='section'>
            <h2>Challenges facing IT Admins</h2>
            <p>IT administrators face several critical challenges when managing service reliability. One of the top concerns is the uncertainty in choosing the right machine learning model and how to evaluate them. With the vast amount of data generated, particularly from system logs, determining the most effective model for anomaly detection can be daunting.
    
    Another significant challenge is the need for selecting and automating actions. Given the complexity of modern IT environments, automating responses to detected anomalies is crucial to maintaining service reliability. Mindmesh addresses these concerns by providing a comprehensive platform for model selection, evaluation, and automation, specifically focusing on processing and analyzing system logs.</p>
            <div class='center'>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/11.Site_Content_Files/11.Images/04.storyset_images/business_case_1.png', use_column_width=True)
    st.markdown("""
    <div class='center'>
        <a href="https://storyset.com/business">Business illustrations by Storyset</a>
    </div>
""", unsafe_allow_html=True)
    
    st.markdown("""
            </div>
        </div>
        <hr>
        <div class='section'>
            <h2>Solution: Mindmesh AI Agents</h2>
            <p>Mindmesh provides a comprehensive solution to the challenges faced by IT administrators. First, it serves as a platform for model selection and evaluation. This addresses the uncertainty of choosing the right machine learning model by providing tools to assess and select the most appropriate models for specific use cases.
    
    Second, Mindmesh is a platform for deploying teams of agents. These autonomous AI agents work together to monitor system logs, detect anomalies, and take proactive actions. By automating these processes, Mindmesh helps ensure continuous and reliable service operations, reducing the burden on IT administrators and minimizing the risk of service disruptions.</p>
            <div class='center'>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/11.Site_Content_Files/11.Images/04.storyset_images/business_case_2.png', use_column_width=True)
    st.markdown("""
    <div class='center'>
        <a href="https://storyset.com/business">Business illustrations by Storyset</a>
    </div>
""", unsafe_allow_html=True)
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Future Vision section
    st.markdown("""
        <hr>
        <div class='section'>
            <h2>Future Vision</h2>
            <p>
            Looking ahead, we aim to expand our AI capabilities to cover more aspects of cloud management, from automated recovery to proactive maintenance. Our goal is to create a fully autonomous cloud environment that can self-heal, adapt, and evolve with minimal human intervention.
            </p>
        </div>
    """, unsafe_allow_html=True)