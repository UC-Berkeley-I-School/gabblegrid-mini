import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Simple Page",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state='collapsed'
)

def main():
    # Display a simple static page
    st.title("Welcome to the Simple Static Page")
    st.write("This is a barebones version of the application, displaying only this static content.")

if __name__ == '__main__':
    main()
