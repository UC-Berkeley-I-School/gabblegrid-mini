import streamlit as st

def display_footer():
    footer_html = """
    <hr style='border: 1px light grey;'>
    <div style='text-align: center;'>
        <a href='mailto:gaurav.narasimhan@berkeley.edu'>
            <i class='fa fa-envelope'></i> gaurav.narasimhan@berkeley.edu
        </a>
        &nbsp;|&nbsp;
        <a href='/?page=privacy_policy' target='_blank'>
            Privacy Policy
        </a>
        &nbsp;|&nbsp;
        <a href='/?page=terms_of_service' target='_blank'>
            Terms of Service
        </a>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    """
    st.markdown(footer_html, unsafe_allow_html=True)
