import streamlit as st

def display_feature_engg_tab():
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

    # Image on the left, text on the right
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/feature_engg_overview_1.gif', use_column_width=True)
        st.markdown("""
            <div class='center'>
                <a href="https://storyset.com/business">Business illustrations by Storyset</a>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='section'>
                <h2>Feature Engineering</h2>
                <p>
                Feature engineering is a crucial step in the data preprocessing pipeline. It involves creating new features or transforming existing ones to enhance the performance of machine learning models. 
                </p>
                <p>
                In this section, we will discuss some of the steps involved in the feature engineering process, which is designed to handle the complexity of log data and extract meaningful patterns for anomaly detection.
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
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/FeatureEngg_Step_1.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Determine Normal and Alert Labels</h2>
            <p>
            In the second step, we label the data as normal or alert based on specific criteria. This labeling is essential for training models to distinguish between regular operations and potential issues.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/FeatureEngg_Step_2.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Aggregate Data Over a 5-Minute Window</h2>
            <p>
            Data is aggregated over sequential 5-minute windows to create a structured dataset. This aggregation helps in capturing temporal patterns and reducing the complexity of the data.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/FeatureEngg_Step_3.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Arrange 5-Minute Sequences for Model Framework</h2>
            <p>
            In this step, we arrange the 5-minute sequences to create a model framework that facilitates training and prediction. This arrangement is critical for ensuring that the model can learn from and predict future events accurately.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/FeatureEngg_Step_4.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Concept of Overlapping and Sequential Windows</h2>
            <p>
            The concept of overlapping and sequential windows is employed to capture a wide range of temporal patterns. Sliding windows overlap, allowing the model to consider overlapping sequences, while sequential windows ensure that each sequence is distinct.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/FeatureEngg_Windows_Concept.png', use_column_width=True)

    st.markdown("""
        <div class='section'>
            <h2>Add Derived Features</h2>
            <p>
            Finally, we add derived features such as event transitions, Shannon entropy, unique events, and the most frequent event. These features provide additional insights into the data and enhance the model's ability to detect anomalies.
            </p>
    """, unsafe_allow_html=True)
    st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/FeatureEngg_Synthetic.png', use_column_width=True)

