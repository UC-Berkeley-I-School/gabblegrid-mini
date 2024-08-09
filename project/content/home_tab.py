import streamlit as st

def display_home_tab():
    # Add the news headline image collage at the top
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/01.Banner/banner_05.png', caption='', use_column_width=True)

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
        <div class='section'>
            <p>
                Developed as a UC Berkeley graduate capstone initiative, GabbleGrid transforms cloud service management with autonomous AI agents that:
                <ol>
                    <li>Analyze vast amounts of log data in real-time</li>
                    <li>Select and optimize machine learning models for precise anomaly detection</li>
                    <li>Execute proactive measures to prevent service disruptions</li>
                </ol>
            </p>
            <p class='grey-text'>GabbleGrid addresses the challenges of service outages in complex IT infrastructures by leveraging the latest advancements in AI and machine learning. Our autonomous agents perform real-time monitoring, predictive maintenance, and proactive anomaly detection, significantly reducing the frequency and impact of service disruptions, thereby ensuring continuous and reliable operations.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='section'>
            <h2>The Cost of Cloud</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Business_Case_0.png', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
            <p class='grey-text'>
            In a cloud-native world, containerized applications and microservices are the norm. While they offer benefits like flexibility and scalability, they're also far more dynamic and complex. Unplanned downtime is significant and pervasive, affecting 80% of organizations within the last three years, with 76% enduring downtime that led to data loss in 2021 alone. The costs associated with these outages include lost productivity, reputational damage, and direct revenue loss.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='section'>
            <h2>Use Case</h2>
            <p>
            A typical use case involves monitoring system logs in real-time to detect anomalies and predict potential outages. Our AI agents analyze log data, identify unusual patterns, and alert relevant teams before an issue escalates, ensuring minimal disruption to services.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # New MVP Concept section
    st.markdown("""
        <hr>
        <div class='section'>
            <h2>MVP Concept</h2>
            <hr>
            <div>
                <h5>Challenges</h5>
                <p>Managing modern cloud environments presents numerous challenges, including the unstructured and unreliable nature of log data, the absence of pretrained models for rapid deployment, and the lack of a comprehensive framework for efficient model testing. Additionally, remediation processes are often manual, making them time-consuming and error-prone.</p>
                <div class='center'>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/MVP_1.png', use_column_width=True)
    st.markdown("""
                </div>
            </div>
            <hr>
            <div class='section'>
                <h5>Solution</h5>
                <p>GabbleGrid addresses these challenges by introducing automated parsing with 'Drain', providing a full suite of pre-trained transformer models, and offering a plug-n-play test harness. This combination enhances reliability, reduces downtime, and streamlines the entire process. Our solution leverages AI-driven insights to automate remediation tasks, reducing the reliance on manual interventions and ensuring faster resolution times.</p>
                <div class='center'>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/MVP_2.png', use_column_width=True)
    st.markdown("""
                </div>
            </div>
            <hr>
            <div class='section'>
                <h5>MVP+</h5>
                <p>Building on the initial solution, MVP+ integrates 324 pre-trained model variants, enabling more robust and diverse model selection. It ensures parsed data is readily available for inference, facilitating real-time decision-making. The GabbleGrid Plug-n-Play system simplifies deployment, allowing for seamless integration into existing infrastructures. Additionally, our agents not only detect anomalies but also generate actionable insights and notifications, working collaboratively with teams to maintain optimal system performance.</p>
                <div class='center'>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/MVP_3.png', use_column_width=True)
    st.markdown("""
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <hr>
        <div class='section'>
            <h2>Future Vision</h2>
            <p>
            Looking ahead, we aim to expand our AI capabilities to cover more aspects of cloud management, from automated recovery to proactive maintenance. Our goal is to create a fully autonomous cloud environment that can self-heal, adapt, and evolve with minimal human intervention.
            </p>
        </div>
    """, unsafe_allow_html=True)