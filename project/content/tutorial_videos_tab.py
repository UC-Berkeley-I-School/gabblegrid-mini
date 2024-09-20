import streamlit as st

def display_tutorial_videos_tab():
    st.markdown("""
        <style>
            .section p, .section h2 {
                color: grey;
            }
            .grey-text {
                color: grey;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns for image and text content, similar to contextual_example_tab.py
    col1, col2 = st.columns([1, 1])

    # Left column for the image and attribution
    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/doc_video_tutorial.gif', use_column_width=True)
        st.markdown("""
            <div class='center'>
                <a href="https://storyset.com/business">Business illustrations by Storyset</a>
            </div>
        """, unsafe_allow_html=True)

    # Right column for the introductory text
    with col2:
        st.markdown("""
            <div class='section'>
                <h2>How To</h2>
                <p>
                    This section contains tutorial videos to guide you through the various features and functionalities of the application.
                    You'll find instructions on navigating the site, using the playground, maintaining the model space, and managing the admin workspace.
                </p>
            </div>
        """, unsafe_allow_html=True)


    # Divider for visual separation
    st.markdown("<hr>", unsafe_allow_html=True)

    # Display video gallery in a 2-column layout similar to playground_main.py
    st.markdown("<h4 style='color: grey;'>Tutorial Videos</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h5 style='color: grey;'>Intro 1</h5>", unsafe_allow_html=True)
        st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/intro_01.mp4')

        st.markdown("<h5 style='color: grey;'>Playground 1</h5>", unsafe_allow_html=True)
        st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/playground_01.mp4')

        st.markdown("<h5 style='color: grey;'>Playground 3</h5>", unsafe_allow_html=True)
        st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/playground_03.mp4')

    with col2:
        st.markdown("<h5 style='color: grey;'>Intro 2</h5>", unsafe_allow_html=True)
        st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/intro_02.mp4')

        st.markdown("<h5 style='color: grey;'>Playground 2</h5>", unsafe_allow_html=True)
        st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/playground_02.mp4')

        st.markdown("<h5 style='color: grey;'>MindSpace & Documentation</h5>", unsafe_allow_html=True)
        st.video('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/videos/mindspace_&_documentatioon_01.mp4')
