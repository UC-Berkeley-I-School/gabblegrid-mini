import streamlit as st

def display_home_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section table, .section td, .section th {
                color: grey; /* Change text color to grey */
            }
            .grey-text {
                color: grey; /* Explicit class for grey text */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <p>
            Developed as a UC Berkeley graduate capstone project, GabbleGrid revolutionizes cloud service management that  employs autonomous agents to:
            <ol>
                <li>Analyze log data</li>
                <li>Select optimal ML models</li>
                <li>Execute preemptive actions</li>
            </ol>
            </p>
            <p class='grey-text'>By predicting and preventing failures, GabbleGrid enhances uptime, reliability, and user experience. We're seeking cloud service provider partnerships to validate our solution in real-world environments.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='section'>
            <h2>The Cost of Cloud</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/Business_Case_1.png', use_column_width=False, width=500)  # Scale to 50%
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

    st.markdown("""
        <div class='section'>
            <h2>MVP (Minimum Viable Product)</h2>
    """, unsafe_allow_html=True)
    
    scale_mvp = st.slider("Scale for MVP image (in %)", 0, 100, 100, key="scale_mvp")
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/MVP_Overview.png', use_column_width=(scale_mvp == 100), width=int(scale_mvp * 10) if scale_mvp != 100 else None)
    
    st.markdown("""
            <p class='grey-text'>
            Our MVP aims to address these challenges through a multi-step process:
            </p>
            <ol class='grey-text'>
                <li>Data Ingestion: Collect and parse system logs.</li>
                <li>Feature Engineering: Create time-based and synthetic features.</li>
                <li>Model Exploration: Evaluate time-series models like LSTM, GRU, and Transformers.</li>
                <li>Model Selection: Choose the best-performing model for deployment.</li>
                <li>Agent Design: Develop AI agents for user proxy, group management, model inference, and communication.</li>
                <li>Tool Design: Design tools for agents to read model results, scan the internet, and compose messages.</li>
                <li>Agent Deployment: Implement agents in a real-world environment to monitor and predict service health.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Future Vision</h2>
            <p>
            Looking ahead, we aim to expand our AI capabilities to cover more aspects of cloud management, from automated recovery to proactive maintenance. Our goal is to create a fully autonomous cloud environment that can self-heal, adapt, and evolve with minimal human intervention.
            </p>
        </div>
    """, unsafe_allow_html=True)
