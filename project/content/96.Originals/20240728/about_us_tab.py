import streamlit as st

def display_about_us_tab():
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
            <h2>About Us</h2>
            <p>
            Welcome to GabbleGrid! We are a team of dedicated professionals committed to revolutionizing cloud service management through autonomous agents and advanced AI technologies. Our mission is to enhance service reliability, uptime, and user experience by predicting and preventing failures in cloud environments. Meet the people behind GabbleGrid:
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/about_us_photo_1.png', use_column_width=True)
        st.markdown("""
            <div class='section'>
                <h2>Gaurav Narasimhan</h2>
                <p><strong>Title:</strong> Engineer</p>
                <p>
                I currently lead a team of data scientists and AI Engineers building AI Autonomous Agents for failure prediction in SaaS cloud services. Concurrently, I am pursuing a graduate degree in Data Science at UC Berkeley, emphasizing practical AI applications and academic research.

As a graduate student at UC Berkeley, I study Natural Language Processing while balancing the demands of a full-time leadership role. I find joy in being both a student and a teacher, an artist, and a technical leader.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/about_us_photo_2.png', use_column_width=True)
        st.markdown("""
            <div class='section'>
                <h2>Oluwafemi Moses Oladejo </h2>
                <p><strong>Title:</strong> Product Manager</p>
                <p>
                I am a senior process engineer at Intel with extensive experience in manufacturing, automation, process development, and optimization. My background includes significant expertise in quantitative research and development. 
                
Currently, I am pursuing a graduate degree in UC Berkeley’s Master of Information and Data Science (MIDS) program. In addition to my professional and academic pursuits, I am an avid sports fan and have a passion for teaching and knowledge exploration..
                </p>
            </div>
        """, unsafe_allow_html=True)
