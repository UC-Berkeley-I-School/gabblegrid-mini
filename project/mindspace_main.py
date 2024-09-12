import streamlit as st

import streamlit as st
from mindspace.essential_reading import display_essential_reading
from mindspace.blog import display_blog

def display_research_insights():
    st.markdown("""
        <style>
            .grey-text { color: grey !important; }
        </style>
    """, unsafe_allow_html=True)

    # Create two-column layout
    col1, col2 = st.columns([1, 1])

    # Left column: display image
    with col1:
        st.image("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/03.Mindspace/mindspace-research-01.gif", use_column_width=True)

    # Right column: display text
    with col2:
        st.markdown("<h3 class='grey-text'>Research Insights & Reflections</h3>", unsafe_allow_html=True)
        
        st.markdown("<p class='grey-text'>In the fast-paced world of Artificial Intelligence, staying up to date with cutting-edge research is key to innovation and progress. This section highlights essential readings and my reflections on key AI advancements and trends.</p>", unsafe_allow_html=True)

        st.markdown("""
        <ul class='grey-text'>
            <li><strong>Stay Current:</strong> Research keeps you informed about the latest breakthroughs in AI models and methodologies.</li>
            <li><strong>Track Trends:</strong> Regular reading helps identify emerging technologies and new areas of growth in AI.</li>
            <li><strong>Gain Perspective:</strong> Understanding recent innovations prevents duplication and sparks fresh approaches in problem-solving.</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("<p class='grey-text'>Engaging with research drives informed action and fosters innovation in the AI landscape.</p>", unsafe_allow_html=True)

def display_mindspace_main():
    display_research_insights()

    tab1, tab2 = st.tabs(["Essential Reading", "Blog"])

    with tab1:
        display_essential_reading()

    with tab2:
        display_blog()

if __name__ == '__main__':
    display_mindspace_main()