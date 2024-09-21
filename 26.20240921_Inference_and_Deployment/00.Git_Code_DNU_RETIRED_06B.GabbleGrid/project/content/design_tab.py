import streamlit as st

def display_design_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2 {
                color: grey; /* Change text color to grey */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>About</h2>
            <p>
            Feature engineering is a crucial step in the data preprocessing pipeline. It involves creating new features or transforming existing ones to enhance the performance of machine learning models. In this section, we will walk through the key steps involved in our feature engineering process, which is designed to handle the complexity of log data and extract meaningful patterns for anomaly detection.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Data Cleansing and Parsing</h2>
            <p>
            The first step involves cleansing the data and parsing it to identify patterns. This step ensures that the data is structured and free of inconsistencies, which is critical for accurate analysis and modeling.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/FeatureEngg_Step_1.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Determine Normal and Alert Labels</h2>
            <p>
            In the second step, we label the data as normal or alert based on specific criteria. This labeling is essential for training models to distinguish between regular operations and potential issues.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/FeatureEngg_Step_2.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Aggregate Data Over a 5-Minute Window</h2>
            <p>
            Data is aggregated over non-overlapping 5-minute windows to create a structured dataset. This aggregation helps in capturing temporal patterns and reducing the complexity of the data.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/FeatureEngg_Step_3.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Arrange 5-Minute Sequences for Model Framework</h2>
            <p>
            In this step, we arrange the 5-minute sequences to create a model framework that facilitates training and prediction. This arrangement is critical for ensuring that the model can learn from and predict future events accurately.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/FeatureEngg_Step_4.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Concept of Sliding and Non-Overlapping Windows</h2>
            <p>
            The concept of sliding and non-overlapping windows is employed to capture a wide range of temporal patterns. Sliding windows overlap, allowing the model to consider overlapping sequences, while non-overlapping windows ensure that each sequence is distinct.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/FeatureEngg_Windows_Concept.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Add Derived Features</h2>
            <p>
            Finally, we add derived features such as event transitions, Shannon entropy, unique events, and the most frequent event. These features provide additional insights into the data and enhance the model's ability to detect anomalies.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project/files/images/FeatureEngg_Synthetic.png', use_column_width=True)
