import streamlit as st

def display_plug_n_play_tab():
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
            Welcome to the Plug-n-Play! This is where the magic happens. Here, you can test the anomaly detection process and see these AI agents in action. 
            The GabbleGrid solution leverages a powerful Transformer model designed specifically for anomaly detection in cloud services.
            </p>
    """, unsafe_allow_html=True)
    
    with st.expander("Read more"):
        st.markdown("""
            <p class='grey-text'>
            The playground takes you through the anomaly detection process, allowing you to see model performance and agents in action. Explore the Playground 
            and witness how these AI agents keep the cloud running smoothly!
            </p>
        </div>
        """, unsafe_allow_html=True)
