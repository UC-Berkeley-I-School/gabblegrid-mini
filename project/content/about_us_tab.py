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
            Welcome to MindMesh! We are a team of dedicated professionals committed to revolutionizing cloud service management through autonomous agents and advanced AI technologies. Our mission is to enhance service reliability, uptime, and user experience by predicting and preventing failures in cloud environments. Meet the people behind MindMesh:
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/about_us_photo_1.png', use_column_width=True)
        st.markdown("""
            <div class='section'>
                <h2>Gaurav Narasimhan</h2>
                <p><strong>Title:</strong> Founder/Engineer</p>
                <p>
I lead a team of data scientists and AI engineers focused on building AI Autonomous Agents for failure prediction in SaaS cloud services. With a recent graduate degree in Data Science from UC Berkeley, my work bridges the gap between practical AI applications and academic research. My experience includes integrating Large Language Models (LLMs) like BERT and GPT to tackle complex challenges, enhancing product capabilities, and developing solutions that drive efficiency and innovation in SaaS engineering.

Throughout my career, I’ve been passionate about the potential of AI and machine learning to transform industries. I’ve worked on projects ranging from autonomous agents to visual document scanning, consistently applying advanced NLP techniques and fostering interdisciplinary collaboration to deliver impactful business solutions. My expertise spans LLMs, machine learning, and autonomous systems, and I continue to explore new ways to apply these technologies to real-world challenges.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/about_us_photo_2.png', use_column_width=True)
        st.markdown("""
            <div class='section'>
                <h2>Mark Butler</h2>
                <p><strong>Title:</strong> Advisor</p>
                <p>
                Senior leader and seasoned Engineering executive with experience in designing and building language-based computer systems. I have extensive experience with automatic classification, information extraction and retrieval, metadata creation, and language generation. I've had the good fortune to spend the last three decades working with various aspects of getting computers to process language to help humans do their work. In these three decades I've watched the fields of computational linguistics and natural language processing make incredible advances from the rule-based Brill Part of Speech Tagger to CRF based statistical SRL parsers to today's neural net language models like BERT and GPT. Because of the length of my experience, I understand the strengths of each of these technological innovations in NLP -- rule-based, statistical, and deep learning. Human language and the ability of computers to process it continue to fascinate me every day.

Specialties: text analytics; text mining; machine learning; information extraction; syntactic/semantic modeling; automated taxonomy development; metadata extraction and systems; information retrieval; open source natural language processing software and system architecture.
                </p>
            </div>
        """, unsafe_allow_html=True)
