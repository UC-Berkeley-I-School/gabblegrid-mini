import streamlit as st
import os
from playground.playground_utils import display_pdf_as_images

# Dictionary mapping folder names to descriptions
folder_descriptions = {
    "01.Illya_25": "Description for Illya_25",
    "02.Placeholder": "Description for Placeholder 2",
    "03.Placeholder": "Description for Placeholder 3",
    "04.Placeholder": "Description for Placeholder 4",
}

def display_essential_reading():
    base_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/01.Papers/'
    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    if st.session_state.page == 'main':
        display_folder_tiles(base_dir)
    else:
        display_pdf_page(base_dir)

def display_folder_tiles(base_dir):
    top_level_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
    st.markdown("""
    <style>
    .stButton > button {
        height: 200px;
        width: 100%;
        color: #808080;  /* Changed to grey */
        border: 2px solid #CCCCCC;
        border-radius: 10px;
        text-align: center;
        word-wrap: break-word;
        white-space: normal;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        font-size: 18px;
    }
    .stButton > button:hover {
        filter: brightness(90%);
    }
    </style>
    """, unsafe_allow_html=True)
    cols = st.columns(4)
    for idx, folder in enumerate(top_level_folders):
        description = folder_descriptions.get(folder, "No description available")
        with cols[idx % 4]:
            if st.button(description, key=f"folder_{idx}", use_container_width=True):
                st.session_state.selected_folder = folder
                st.session_state.selected_description = description
                st.session_state.page = 'pdf'
                st.experimental_rerun()

def display_pdf_page(base_dir):
    st.markdown("""
    <style>
    .stButton > button {
        background-color: #E6F3FF;  /* Pastel blue */
        color: #808080;  /* Grey text */
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 25px;  /* Rounded corners */
        transition: background-color 0.3s ease;  /* Smooth transition for hover effect */
    }
    .stButton > button:hover {
        background-color: #D1E8FF;  /* Slightly darker pastel blue for hover effect */
    }
    </style>
    """, unsafe_allow_html=True)

    # Display the description instead of the folder name
    st.markdown(f"""
    <h3 style="color: grey;">
        {st.session_state.selected_description}
    </h3>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Folders"):
        st.session_state.page = 'main'
        st.experimental_rerun()
    
    folder_path = os.path.join(base_dir, st.session_state.selected_folder)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    # Sort the PDF files alphabetically
    sorted_pdf_files = sorted(pdf_files, key=lambda x: x.lower())
    
    for pdf_file in sorted_pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        display_name = os.path.splitext(pdf_file)[0]  # Remove the file extension
        
        with st.expander(display_name):
            display_pdf_as_images(file_path, display_name, key_prefix=f"essential_reading_{display_name}")
            
if __name__ == "__main__":
    display_essential_reading()
