import streamlit as st
from content.transformers_tab import display_transformers_tab
from content.tech_tab import display_tech_tab
from content.feature_engg_tab import display_feature_engg_tab
from content.contextual_example_tab import display_contextual_example_tab
from content.product_vision_tab import display_product_vision_tab

def display_documentation_tab():
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(["Vision", "Feature Engg", "Tech Stack",  "Architecture", "Contextual Example", ])

    with subtab1:
        display_product_vision_tab()
    
    with subtab2:
        display_feature_engg_tab()

    with subtab3:
        display_tech_tab()    
    
    with subtab4:
        display_transformers_tab()

    with subtab5:
        display_contextual_example_tab()

