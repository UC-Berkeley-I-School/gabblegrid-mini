import streamlit as st
import pandas as pd
import os

def display_models_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section table, .section td, .section th, h5, h6 {
                color: grey; /* Change text color to grey */
            }
            .grey-text {
                color: grey; /* Explicit class for grey text */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
            .input-group {
                display: flex;
                flex-direction: column;
                margin-bottom: 1rem;
            }
            .input-label {
                margin-bottom: 0.5rem;
                font-weight: bold;
                color: #444;
            }
            .input-field {
                padding: 0.5rem;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .tab-container {
                margin-top: 1rem;
                padding: 1rem;
                background-color: #f9f9f9;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .tab-header {
                font-weight: bold;
                color: #333;
                margin-bottom: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Models</h2>
            <p>Welcome to the Models tab! Here you can test the anomaly detection process using our pre-trained models or upload your own models for inference testing.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("Read more about model categories"):
        st.markdown("""
            <div class='section'>
                <ul>
                    <li><strong>Sequential Windows</strong>
                        <ul>
                            <li><strong>Seeded/Open Source:</strong> Pre-trained models provided by MindMesh.</li>
                            <li><strong>User Created/Uploaded:</strong> Models uploaded by users.</li>
                        </ul>
                    </li>
                    <li><strong>Overlapping Windows</strong>
                        <ul>
                            <li><strong>Seeded/Open Source:</strong> Pre-trained models provided by MindMesh.</li>
                            <li><strong>User Created/Uploaded:</strong> Models uploaded by users.</li>
                        </ul>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    def display_available_models(csv_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.dataframe(df)
        else:
            st.markdown("No models available.")

    # Upload models
    def upload_model(upload_dir, tracker_path, category, model_type, key):
        # Check if the user is authorized
        if st.session_state.get('user', {}).get('email') != "gaurav.narasimhan@berkeley.edu":
            st.warning("You do not have permission to upload models. Please contact the administrator for access.")
            return
    
        uploaded_file = st.file_uploader(f"Upload your {category} - {model_type} model", type=['pt', 'pth'], key=key)
        if uploaded_file:
            file_size = len(uploaded_file.getvalue())
            st.write(f"File size: {file_size / (1024 * 1024):.2f} MB")  # Show file size
            if file_size > 200 * 1024 * 1024:  # Check if file size exceeds 200 MB
                st.error("File size exceeds the maximum limit of 200 MB.")
                return
    
            # Prompt user to enter details for the tracker CSV
            details = {}
            st.markdown("<h3>Enter Model Details</h3>", unsafe_allow_html=True)
            required_fields = ['Max Events', 'Input Length', 'Hidden Size', 'Dropout', 'Num Layers', 'Gap', 'Prediction Period', 'Num Epochs']
            optional_fields = ['Precision', 'Recall', 'Accuracy', 'F1', 'TN', 'FP', 'FN', 'TP']
    
            for field in required_fields:
                details[field] = st.text_input(f"Enter value for {field}", key=f"input_{field}_{key}", help=f"Required field: {field}")
    
            for field in optional_fields:
                details[field] = st.text_input(f"Enter value for {field} (optional)", key=f"input_{field}_{key}", help=f"Optional field: {field}")
    
            if st.button("Submit", key=f"submit_{key}"):
                with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded {uploaded_file.name} to {upload_dir}")
    
                # Generate the run number
                if os.path.exists(tracker_path):
                    tracker_df = pd.read_csv(tracker_path)
                    run_number = tracker_df['Run'].max() + 1
                else:
                    tracker_df = pd.DataFrame(columns=['Run'] + required_fields + optional_fields)
                    run_number = 1
    
                details['Run'] = run_number
    
                new_entry = pd.Series(details)
                tracker_df = pd.concat([tracker_df, new_entry.to_frame().T], ignore_index=True)
                tracker_df.to_csv(tracker_path, index=False)
                st.success(f"Updated tracker for {category} - {model_type}")
        else:
            st.info("Please upload a model file to proceed.")
    

    
    st.markdown("""
        <div class='section'>
            <h2>Available Models</h2>
    """, unsafe_allow_html=True)

    available_tabs = st.tabs(["Sequential Windows", "Overlapping Windows"])

    with available_tabs[0]:
        st.markdown("<div class='tab-container'><h5>Seeded/Open Source</h5></div>", unsafe_allow_html=True)
        display_available_models("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv")

    with available_tabs[1]:
        st.markdown("<div class='tab-container'><h5>Seeded/Open Source</h5></div>", unsafe_allow_html=True)
        display_available_models("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_Overlapping_Consl_144.csv")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>User Created Models</h2>
    """, unsafe_allow_html=True)

    user_created_tabs = st.tabs(["Sequential Windows", "Overlapping Windows"])

    with user_created_tabs[0]:
        st.markdown("<div class='tab-container'><h5>Upload Your Model</h5></div>", unsafe_allow_html=True)
        upload_model("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/20240729_No_Overlap_Plug_n_Play", "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240729_Transformers_NonOverlapping_User_Cat1.csv", "Sequential Windows", "User Created/Uploaded", key="user_category_1")
        st.markdown("<div class='tab-container'><h5>User Created</h5></div>", unsafe_allow_html=True)
        display_available_models("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240729_Transformers_NonOverlapping_User_Cat1.csv")

    with user_created_tabs[1]:
        st.markdown("<div class='tab-container'><h5>Upload Your Model</h5></div>", unsafe_allow_html=True)
        upload_model("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/20240729_Yes_Overlap_Plug_n_Play", "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240729_Transformers_NonOverlapping_User_Cat2.csv", "Overlapping Windows", "User Created/Uploaded", key="user_category_2")
        st.markdown("<div class='tab-container'><h5>User Created</h5></div>", unsafe_allow_html=True)
        display_available_models("/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240729_Transformers_NonOverlapping_User_Cat2.csv")

    st.markdown("</div>", unsafe_allow_html=True)