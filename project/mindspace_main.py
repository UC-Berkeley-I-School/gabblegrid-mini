# mindspace_main.py

import streamlit as st
from mindspace.essential_reading import display_essential_reading
from mindspace.blog import display_blog

def display_why_reading_matters():
    st.markdown("""
        <style>
            .grey-text { color: grey !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h3 class='grey-text'>Why Reading Matters</h3>", unsafe_allow_html=True)
    
    st.markdown("<p class='grey-text'>In the rapidly evolving field of Artificial Intelligence, staying current with research is not just beneficial—it's essential. The MindSpace tab is designed to facilitate this crucial aspect of AI development and understanding.</p>", unsafe_allow_html=True)

    # Wrap Essential Reading, Blog, and The Importance of Research in AI in a single expander
    with st.expander("Learn More"):
        st.markdown("<h5 class='grey-text'>Essential Reading</h5>", unsafe_allow_html=True)
        st.markdown("<p class='grey-text'>The Essential Reading section contains a curated collection of research papers, organized into different topics. These papers represent the cornerstone of modern AI development and are either part of my completed reading list or are high-priority items on my to-read list.</p>", unsafe_allow_html=True)

        st.markdown("<h5 class='grey-text'>Blog</h5>", unsafe_allow_html=True)
        st.markdown("<p class='grey-text'>The Blog section offers insights, reflections, and discussions on various AI topics, serving as a platform for deeper exploration of concepts encountered in the research papers.</p>", unsafe_allow_html=True)

        st.markdown("<h5 class='grey-text'>The Importance of Research in AI</h5>", unsafe_allow_html=True)
        st.markdown("<p class='grey-text'>AI is fundamentally a research-driven field. The pace of innovation is incredibly rapid, with new techniques, models, and paradigms emerging constantly. Staying abreast of these developments is crucial for several reasons:</p>", unsafe_allow_html=True)

        st.markdown("""
        <ul class='grey-text'>
            <li><strong>Understanding State-of-the-Art:</strong> Research papers provide insight into the most advanced techniques and models currently available.</li>
            <li><strong>Identifying Trends:</strong> Regular reading helps in recognizing emerging trends and potential future directions in AI.</li>
            <li><strong>Problem-Solving:</strong> Many papers present novel solutions to complex problems, which can inspire new approaches in your own work.</li>
            <li><strong>Avoiding Reinvention:</strong> Knowledge of existing research prevents unnecessary duplication of efforts.</li>
            <li><strong>Interdisciplinary Insights:</strong> AI research often intersects with other fields, offering valuable cross-disciplinary perspectives.</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("<p class='grey-text'>By engaging with this content, you're not just consuming information—you're participating in the global conversation that's shaping the future of AI.</p>", unsafe_allow_html=True)

def display_mindspace_main():
    display_why_reading_matters()

    tab1, tab2 = st.tabs(["Essential Reading", "Blog"])

    with tab1:
        display_essential_reading()

    with tab2:
        display_blog()

if __name__ == '__main__':
    display_mindspace_main()