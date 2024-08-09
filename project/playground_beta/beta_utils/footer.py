import streamlit as st

def display_footer():
    footer_html = """
    <hr style='border: 1px light grey;'>
    <div style='text-align: center;'>
        <i class='fa fa-envelope'></i> gaurav.narasimhan@berkeley.edu
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    """
    st.markdown(footer_html, unsafe_allow_html=True)
