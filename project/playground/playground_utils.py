# File: playground/playground_utils.py

import streamlit as st
from pdf2image import convert_from_path
import os

def display_pdf_as_images(pdf_path, display_name, key_prefix=None):
    images = convert_from_path(pdf_path, first_page=0, last_page=1)
    if images:
        st.image(images[0], caption=f'{display_name} - Page 1', use_column_width=True)
        with open(pdf_path, "rb") as f:
            st.download_button(label=f"Download", data=f, file_name=os.path.basename(pdf_path), key=f'{key_prefix}_download' if key_prefix else None)
