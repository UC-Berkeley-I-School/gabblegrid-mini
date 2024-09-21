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
            <h2>About MindMesh</h2>
            <p>
            MindMesh continues the research-driven approach developed during a UC Berkeley graduate capstone, focusing on advancing AI agents for enterprise use cases. By exploring how autonomous agents can improve system reliability and performance, MindMesh aims to push the boundaries of AI in complex IT environments.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/10.About_Us/about_us_photo_1.png', use_column_width=True)
        st.markdown("""
            <div class='section'>
                <h2>Gaurav Narasimhan</h2>
                <p><strong>Title:</strong> Founder/Engineer</p>
                <p>
I specialize in building AI autonomous agents for failure prediction in cloud services, with a focus on applied research and bridging the gap between academic insights and practical AI applications. With a recent graduate degree in Data Science from UC Berkeley and as a current graduate student at Stanford Engineering (CS), I am committed to continuous learning and advancing AI technologies in real-world scenarios.

My career has centered around leveraging AI and machine learning to drive innovation and efficiency. I’ve led projects that span autonomous agents, anomaly detection, and NLP systems, continually exploring new ways to apply AI technologies to practical challenges. As a research-oriented engineer, I strive to push the boundaries of what’s possible in AI, always with a learner's mindset.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <a href="https://www.linkedin.com/in/gauravnarasimhan/" target="_blank">
                <button style="background-color: #0077B5; color: white; padding: 10px 20px; border: none; border-radius: 20px; cursor: pointer;">
                    Connect on LinkedIn
                </button>
            </a>
        """, unsafe_allow_html=True)

        # # Use HTML in markdown to create a clickable image
        # st.markdown(
        #     f"""
        #     <a href="https://www.linkedin.com/in/gauravnarasimhan/" target="_blank">
        #         <img src="/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/10.About_Us/linkedin_connect_1.png" 
        #         alt="Connect with me on LinkedIn" style="width:200px;">
        #     </a>
        #     """, 
        #     unsafe_allow_html=True
        # )

    
############################# REVERT ONCE MARK CONFIRMS #########################
    
#     with col2:
#         st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/10.About_Us/about_us_photo_2.png', use_column_width=True)
#         st.markdown("""
#             <div class='section'>
#                 <h2>Mark Butler</h2>
#                 <p><strong>Title:</strong> Advisor</p>
#                 <p>
#                 Senior leader and seasoned Engineering executive with experience in designing and building language-based computer systems. I have extensive experience with automatic classification, information extraction and retrieval, metadata creation, and language generation. I've had the good fortune to spend the last three decades working with various aspects of getting computers to process language to help humans do their work. In these three decades I've watched the fields of computational linguistics and natural language processing make incredible advances from the rule-based Brill Part of Speech Tagger to CRF based statistical SRL parsers to today's neural net language models like BERT and GPT. Because of the length of my experience, I understand the strengths of each of these technological innovations in NLP -- rule-based, statistical, and deep learning. Human language and the ability of computers to process it continue to fascinate me every day.

# Specialties: text analytics; text mining; machine learning; information extraction; syntactic/semantic modeling; automated taxonomy development; metadata extraction and systems; information retrieval; open source natural language processing software and system architecture.
#                 </p>
#             </div>
#         """, unsafe_allow_html=True)

####################################################################################################
    
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/10.About_Us/about_us_photo_blank.png', use_column_width=True)
        st.markdown("""
            <div class='section'>
                <h2>Will be updated shortly</h2>
                <p><strong>Will be updated shortly:</strong> Advisor</p>
                <p>
                Will be updated shortly 
                </p>
            </div>
        """, unsafe_allow_html=True)
