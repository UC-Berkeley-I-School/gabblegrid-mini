import streamlit as st
from content.transformers_tab import display_transformers_tab
from content.tech_tab import display_tech_tab
from content.design_tab import display_design_tab

def display_documentation_tab():
    subtab1, subtab2, subtab3 = st.tabs(["Transformers", "Tech", "Design"])

    with subtab1:
        display_transformers_tab()

    with subtab2:
        display_tech_tab()

    with subtab3:
        display_design_tab()
