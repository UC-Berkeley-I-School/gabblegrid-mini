import streamlit as st
import os
from pdf2image import convert_from_path

def display_pdf_as_images(file_path, display_name):
    try:
        images = convert_from_path(file_path, first_page=0, last_page=1)
        for image in images:
            st.image(image, caption=f"{display_name} - Page 1", use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred while displaying '{display_name}': {e}")

def display_essential_reading():
    base_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/01.Papers/'

    # Add a 3-1 column layout for the text and image
    col1, col2 = st.columns([3, 1])
    
    # Left column: display text summarizing the contents of the 5 folders
    with col1:
        st.markdown("""
            <h5 class='grey-text'>A Personal and Curated AI Journey</h5>
            <p class='grey-text'>
                This section serves as much as my personal reading list as it is a curated selection for others. I've read some of these foundational and advanced papers, while several more remain on my to-do list. These papers cover key areas of Artificial Intelligence and Machine Learning such as 
                <strong>Annotated Transformer</strong>, 
                <strong>Recurrent Neural Networks</strong>, 
                <strong>LSTM Networks</strong>, 
                and <strong>Attention Mechanisms</strong>.
            </p>
            <p class='grey-text'>
                Whether you're just getting started or diving deeper into AI research, this collection offers a rich set of resources to help you expand your knowledge and keep pace with advancements in the field. Join me on this journey of continuous learning and exploration.
            </p>
        """, unsafe_allow_html=True)
    
    # Right column: display image
    with col2:
        st.image("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/03.Mindspace/mindspace-essential-reading-01.gif", use_column_width=True)
        st.markdown("""
        <div class='center'>
            <a href="https://storyset.com/business">Business illustrations by Storyset</a>
        </div>
    """, unsafe_allow_html=True)

    display_main_page(base_dir)

def display_main_page(base_dir):
    parent_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.') and not f.endswith('_checkpoints')])
    
    if 'selected_folder' not in st.session_state or st.session_state.selected_folder is None:
        tabs = st.tabs(parent_folders)
        
        for i, folder in enumerate(parent_folders):
            with tabs[i]:
                folder_path = os.path.join(base_dir, folder)
                if folder == "11.Staging":
                    display_pdf_files(folder_path)
                else:
                    sub_folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and not f.startswith('.') and not f.endswith('_checkpoints')])
                    display_sub_folder_tiles(folder, folder_path, sub_folders)
    else:
        display_sub_folder_page(os.path.join(base_dir, st.session_state.selected_folder), st.session_state.selected_folder)

def display_sub_folder_tiles(parent_folder, folder_path, sub_folders):
    st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"] [data-testid="column"] .stButton > button {
        height: 200px !important;
        width: 200px !important;
        color: #808080 !important;
        border: none !important;
        border-radius: 10px !important;
        text-align: center !important;
        word-wrap: break-word !important;
        white-space: normal !important;
        padding: 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 10px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4n+1) .stButton > button { background-color: #E6F3FF !important; }
    [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4n+2) .stButton > button { background-color: #FFEEE6 !important; }
    [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4n+3) .stButton > button { background-color: #FFF7E6 !important; }
    [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4n+4) .stButton > button { background-color: #E6FFE6 !important; }
    
    [data-testid="stHorizontalBlock"] [data-testid="column"] .stButton > button:hover {
        filter: brightness(90%) !important;
        transform: scale(1.05) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    for idx, sub_folder in enumerate(sorted(sub_folders)):
        with cols[idx % 4]:
            # Store state only if the folder is changed to avoid rerun
            if st.button(sub_folder, key=f"{parent_folder}_{sub_folder}_button", use_container_width=True):
                if st.session_state.get("selected_sub_folder") != sub_folder:
                    st.session_state.selected_folder = parent_folder
                    st.session_state.selected_sub_folder = sub_folder
                    st.experimental_rerun()

def display_sub_folder_page(folder_path, parent_folder):
    st.markdown(f"""
    <p style="color: #007bff; font-size: 16px;">
        {parent_folder} > {st.session_state.selected_sub_folder}
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .stButton > button {
        background-color: #FFB3BA !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Main", key=f"back_button_{parent_folder}_{st.session_state.selected_sub_folder}"):
        st.session_state.selected_folder = None
        st.session_state.selected_sub_folder = None
        st.experimental_rerun()

    sub_folder_path = os.path.join(folder_path, st.session_state.selected_sub_folder)
    if os.path.exists(sub_folder_path):
        display_pdf_files(sub_folder_path)
    else:
        st.error(f"Folder not found: {sub_folder_path}")

def display_pdf_files(folder_path):
    try:
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        sorted_pdf_files = sorted(pdf_files, key=lambda x: x.lower())

        if not pdf_files:
            st.warning(f"No PDF files found in {folder_path}")
            return

        for pdf_file in sorted_pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            display_name = os.path.splitext(pdf_file)[0]
            
            with st.expander(display_name, expanded=False):
                display_pdf_as_images(file_path, display_name)
                
                # Add some vertical space
                st.write("")
                
                # Place download button at the bottom right
                col1, col2, col3 = st.columns([1, 1, 1])
                with col3:
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label="Download PDF",
                            data=file,
                            file_name=pdf_file,
                            mime="application/pdf",
                            key=f"download_{pdf_file}"
                        )
    except FileNotFoundError:
        st.error(f"Folder not found: {folder_path}")
    except Exception as e:
        st.error(f"An error occurred while displaying files from {folder_path}: {str(e)}")
        
if __name__ == "__main__":
    display_essential_reading()