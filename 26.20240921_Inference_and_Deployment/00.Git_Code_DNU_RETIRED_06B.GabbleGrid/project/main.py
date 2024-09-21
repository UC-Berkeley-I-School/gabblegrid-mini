import streamlit as st
import time
import asyncio
import json
from datetime import datetime
from autogen.cache import Cache
from content.home_tab import display_home_tab
from content.playground_tab import display_playground_tab
from content.design_tab import display_design_tab
from content.tech_tab import display_tech_tab
from content.models_tab import display_models_tab
from content.about_us_tab import display_about_us_tab
import autogen
from utils.footer import display_footer

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="GabbleGrid",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state='collapsed'
    )

    # Load the CSS file
    load_css('utils/styles.css')

    # Add the news headline image collage
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/10.Image_and_Static_Content/20240720_banner.jpeg', caption='', use_column_width=True)
    
    # Title
    st.markdown("""
        <div class='section header'>
            <div class='title'>GabbleGrid: Self-Healing Clouds with AI Agents</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Add horizontal tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Home", "Playground", "Design", "Tech", "Models", "About Us"])
    
    with tab1:
        display_home_tab()
        display_footer()
    
    with tab2:
        display_playground_tab()
        display_footer()
    
    with tab3:
        display_design_tab()
        display_footer()
    
    with tab4:
        display_tech_tab()
        display_footer()
    
    with tab5:
        display_models_tab()
        display_footer()
    
    with tab6:
        display_about_us_tab()
        display_footer()

if __name__ == '__main__':
    main()
