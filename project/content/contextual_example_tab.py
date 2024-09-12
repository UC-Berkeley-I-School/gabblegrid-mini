import streamlit as st

def display_contextual_example_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section table, .section td, .section th {
                color: grey; /* Change text color to grey */
            }
            .grey-text {
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

#########################################################################################################
    
    # # Section 3
    # Create two columns for image and text content
    col1, col2 = st.columns([1, 1])

    # Left column for the image and attribution
    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/documentation_example_1.gif', use_column_width=True)
        st.markdown("""
            <div class='center'>
                <a href="https://storyset.com/business">Business illustrations by Storyset</a>
            </div>
        """, unsafe_allow_html=True)

    # Right column for the introductory text
    with col2:
        st.markdown("""
            <div class='section'>
                <h2>Disk I/O Errors</h2>
                <p>
                    In this section, we explore an example case of managing disk I/O errors and how a team of agents can be modeled to address the issue effectively. 
                    The goal is to showcase how different agents work together to detect, analyze, and resolve a critical system failure.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # New div for the detailed agent team content, placed after the image and introductory text
    st.markdown("""
        <div class='section'>
            <h2>Agent Team for Disk I/O Error Resolution</h2>
            <p>
                The team of agents working on this case includes:
                <ul>
                    <li><strong>Detection Agent:</strong> Identifies a recurring disk I/O error in the logs, indicating potential hardware failure.</li>
                    <li><strong>Analysis Agent:</strong> Confirms a failing disk on say, Node 42, providing diagnostic insights into the problem.</li>
                    <li><strong>Action Agent:</strong> Schedules an immediate disk replacement and workload migration to prevent downtime.</li>
                    <li><strong>Execution Agent:</strong> Coordinates with the data center team to replace the disk, ensuring no data loss during the process.</li>
                    <li><strong>Notification Agent:</strong> Sends an update to administrators and updates the monitoring dashboard with the resolution status.</li>
                </ul>
            </p>
        </div>
    """, unsafe_allow_html=True)

#########################################################################################################
    
    # Section 4 - Simplified Content for Disk I/O Error Identification
    st.markdown("""
        <div class='section'>
            <h2>The challenges of identifying the error in advance</h2>
            <p>
                Identifying disk I/O errors is challenging due to the diversity of hardware, configurations, and the complexity of error logs. Specialized detection models are required to filter out anomalies and pinpoint the root cause of issues. This process involves filtering, classification, and in-depth analysis to ensure accurate identification and resolution.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display image related to Disk I/O errors
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/contextual_example_1.png', use_column_width=True)
    
    # Divider for visual separation
    st.markdown("<hr>", unsafe_allow_html=True)

    
#########################################################################################################
    
    # Section 5 - Simplified Content for Disk I/O Error Identification
    st.markdown("""
        <div class='section'>
            <h2>Key Aspects of Disk I/O Error Identification</h2>
            <p>
                Identifying disk I/O errors requires understanding key aspects such as hardware diversity, specialized detection models, and the need for fine-tuning based on system characteristics. 
                Further detailed analysis and consideration of contextual factors ensure accurate error identification and resolution.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display image related to contextual example
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/contextual_example_2.png', use_column_width=True)
    
    # Divider for visual separation
    st.markdown("<hr>", unsafe_allow_html=True)

#########################################################################################################
    
    # Section 5 - Simplified Content for Disk I/O Error Resolution
    st.markdown("""
        <div class='section'>
            <h2>Resolution</h2>
            <p>
                Resolving disk I/O errors involves choosing from multiple potential actions, making context-sensitive decisions, and ensuring proper coordination and communication among teams.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display image related to Disk I/O error resolution
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/contextual_example_3.png', use_column_width=True)
    
    # Divider for visual separation
    st.markdown("<hr>", unsafe_allow_html=True)


#########################################################################################################
    
    # Section 7 - Simplified Content for Disk I/O Error Resolution
    st.markdown("""
        <div class='section'>
            <h2>Resolution Actions</h2>
            <p>
                Resolving disk I/O errors involves several key actions, ranging from diagnostic checks and software adjustments to hardware replacements. Each action has its own considerations based on the severity and nature of the issue.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display image related to Disk I/O error resolution actions
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/contextual_example_4.png', use_column_width=True)
    
    # Divider for visual separation
    st.markdown("<hr>", unsafe_allow_html=True)


#########################################################################################################

    # Section 8 - Simplified Content for Disk I/O Error Resolution Paths
    st.markdown("""
        <div class='section'>
            <h2>Resolution Paths</h2>
            <p>
                Resolving disk I/O errors can involve multiple approaches, including RAID rebuilds, switching to backup disks, verifying alerts, and collaborating with vendors. Each path has its own considerations, from ensuring data redundancy to minimizing operational impact.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display image related to Disk I/O resolution paths
    col1, col2, col3 = st.columns([1, 40, 1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/contextual_example_5.png', use_column_width=True)
    
    # Divider for visual separation
    st.markdown("<hr>", unsafe_allow_html=True)

