import streamlit as st
import os
from pdf2image import convert_from_path

# Display PDF as images and handle errors gracefully
def display_pdf_as_images(file_path, display_name):
    """Attempt to convert PDF to images, handle errors gracefully."""
    try:
        st.write(f"Displaying PDF: {display_name}")
        images = convert_from_path(file_path, first_page=0, last_page=1)
        for image in images:
            st.image(image, caption=f"{display_name} - Page 1")
    except Exception as e:
        st.error(f"An error occurred while displaying '{display_name}': {e}")

def display_essential_reading():
    base_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/mindspace/01.Papers/'

    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    if st.session_state.page == 'main':
        # List the parent folders
        parent_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.') and not f.endswith('_checkpoints')])

        # Create a tab for each parent folder
        tabs = st.tabs(parent_folders)
        
        # Loop through each parent folder and display its contents
        for i, folder in enumerate(parent_folders):
            with tabs[i]:
                folder_path = os.path.join(base_dir, folder)
                sub_folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and not f.startswith('.') and not f.endswith('_checkpoints')])

                # If sub-folders exist, display them as buttons
                if sub_folders:
                    display_sub_folder_tiles(folder, folder_path, sub_folders)
                else:
                    # If there are no sub-folders, display the files in the parent folder
                    display_files_in_folder(folder_path)
    else:
        display_pdf_page(base_dir)

# Updated function to display sub-folders in a 4-column grid with styled buttons
def display_sub_folder_tiles(parent_folder, folder_path, sub_folders):
    """Display sub-folders as buttons in a 4-column grid."""
    st.markdown("""
    <style>
    .stButton > button {
        height: 60px;
        width: 200px;
        color: #808080;
        border: none;
        border-radius: 10px;
        text-align: center;
        word-wrap: break-word;
        white-space: normal;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        filter: brightness(90%);
        transform: scale(1.05);
    }
    .stButton:nth-of-type(4n+1) > button { background-color: #E6F3FF; }  /* Light Blue */
    .stButton:nth-of-type(4n+2) > button { background-color: #FFEEE6; }  /* Light Red */
    .stButton:nth-of-type(4n+3) > button { background-color: #FFF7E6; }  /* Light Orange */
    .stButton:nth-of-type(4n+4) > button { background-color: #E6FFE6; }  /* Light Green */
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    for idx, sub_folder in enumerate(sorted(sub_folders)):
    # for idx, sub_folder in enumerate(sub_folders):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        with cols[idx % 4]:
            if st.button(sub_folder, key=f"{parent_folder}_sub_folder_button_{idx}", use_container_width=True):
                st.session_state.selected_folder = os.path.join(parent_folder, sub_folder)
                st.session_state.selected_description = sub_folder
                st.session_state.page = 'pdf'
                st.experimental_rerun()

def display_files_in_folder(folder_path):
    """Display the PDF files in a folder."""
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    # Sort the PDF files alphabetically
    sorted_pdf_files = sorted(pdf_files, key=lambda x: x.lower())
    
    for idx, pdf_file in enumerate(sorted_pdf_files):
        file_path = os.path.join(folder_path, pdf_file)
        display_name = os.path.splitext(pdf_file)[0]  # Remove the file extension
        
        with st.expander(display_name, expanded=False):
            display_pdf_as_images(file_path, display_name)

# Retained original style for buttons and folder tiles from reference code
def display_folder_tiles(base_dir):
    top_level_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
    st.markdown("""
    <style>
    .stButton > button {
        height: 60px;
        width: 200px;
        color: #808080;
        border: none;
        border-radius: 10px;
        text-align: center;
        word-wrap: break-word;
        white-space: normal;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        filter: brightness(90%);
        transform: scale(1.05);
    }
    .stButton:nth-of-type(4n+1) > button { background-color: #E6F3FF; }  /* Red */
    .stButton:nth-of-type(4n+2) > button { background-color: #E6F3FF; }  /* Teal */
    .stButton:nth-of-type(4n+3) > button { background-color: #E6F3FF; }  /* Blue */
    .stButton:nth-of-type(4n+4) > button { background-color: #E6F3FF; }  /* Light Salmon */
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
    .stDownloadButton > button {
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
    .stDownloadButton > button:hover {
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
    
    # Split the selected folder path into parent and sub folder
    parent_folder, sub_folder = os.path.split(st.session_state.selected_folder)
    folder_path = os.path.join(base_dir, parent_folder, sub_folder)
    
    try:
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        
        # Sort the PDF files alphabetically
        sorted_pdf_files = sorted(pdf_files, key=lambda x: x.lower())
        
        for pdf_file in sorted_pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            display_name = os.path.splitext(pdf_file)[0]  # Remove the file extension
            
            with st.expander(display_name):
                display_pdf_as_images(file_path, display_name)
                
                # Add some vertical space
                st.write("")
                
                # Place download button at the bottom left with custom styling
                with open(file_path, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name=pdf_file,
                        mime="application/pdf"
                    )

    except FileNotFoundError:
        st.error(f"Folder not found: {folder_path}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    display_essential_reading()
