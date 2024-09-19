import streamlit as st


def display_transformers_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section table, .section td, .section th {
                color: grey; /* Change text color to grey */
            }
            .grey-text {
                color: grey; /* Explicit class for grey text */
            }
            details summary {
                color: grey; /* Change 'read more' text color to grey */
            }
            /* Remove grey background from slider end labels */
            .stSlider .stMarkdown {
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns: text on the left, image on the right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div class='section'>
                <h2>Current Standard: Transformers</h2>
                <p>
                    The encoder-only transformer with a binary classification head is particularly suitable for the use case. 
                    It focuses on capturing long-range dependencies in log sequences while simplifying the model by excluding the decoder. 
                    This reduces complexity and enhances the model's ability to process large-scale log data efficiently.
                </p>
                <p>
                    Several other architectures have also been evaluated, including LSTMs (Long Short-Term Memory networks) and GRUs (Gated Recurrent Units), 
                    both of which are effective in capturing sequential dependencies. Isolation Forest and LogBERT are additional methods specifically 
                    designed for anomaly detection in time series data. Each of these architectures offers unique strengths, such as the ability of 
                    LSTMs and GRUs to capture temporal dynamics and the ability of LogBERT to handle complex log patterns.
                </p>
                <p>
                    The decision to use Transformers is not final, and we are currently re-evaluating other architectures in our ongoing efforts 
                    to optimize model performance. Techniques such as Autoencoders, CNNs, and newer attention-based models are also under consideration. 
                    The goal remains to find the most effective architecture that balances performance, scalability, and ease of deployment in production.
                </p>
            </div>
        """, unsafe_allow_html=True)    


    
    # Right column for the image and attribution
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/04.Documentation/documentation_transformer_1.gif', use_column_width=True)
        st.markdown("""
            <div class='center'>
                <a href="https://storyset.com/business">Business illustrations by Storyset</a>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Placeholder 1
    st.markdown("""
        <div class='section'>
            <h2>Rationale</h2>
            <p>
            The Transformer model was chosen for its ability to handle complex data challenges. It is particularly effective for:
            </p>
            <details>
                <summary>Read more</summary>
                <p>
            <ul>
                <li><strong>Long-range Dependency Handling:</strong> Transformers capture long-term dependencies in log sequences more effectively than traditional RNNs, which is crucial for accurate anomaly detection.</li>
                <li><strong>Parallelization:</strong> The ability to process log sequences in parallel makes Transformers well-suited for big data problems, significantly speeding up training and inference times.</li>
                <li><strong>Positional Encoding:</strong> By maintaining the order of log entries, positional encoding ensures that the temporal nature of the data is preserved, which is vital for time series analysis.</li>
                <li><strong>Flexibility:</strong> The versatile architecture of Transformers allows them to be adapted for various use cases beyond the current scope, providing a robust solution for future challenges.</li>
            </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Model_Selection_05.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Placeholder 2
    st.markdown("""
        <div class='section'>
            <h2>Model Architecture</h2>
            <p>
            The Encoder Only Model with Classification Head uses only the encoder for processing sequential log data. It transforms logs into high-dimensional embeddings with positional information, captures long-range dependencies in log sequences, and outputs binary classifications.
            </p>
            <details>
                <summary>Read more</summary>
                <p>
                The model architecture involves several key components: 
                <ul>
                    <li>Embedding Layer: Projects input features into a higher-dimensional space.</li>
                    <li>Positional Encoding: Adds positional information to the embeddings.</li>
                    <li>Transformer: Captures long-range dependencies in the log sequences.</li>
                    <li>Classification Head: Maps the transformer's output to a binary classification.</li>
                    <li>Forward Pass: Processes the embeddings and outputs binary classification.</li>
                </ul>
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Model_Selection_04.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Placeholder 4
    st.markdown("""
        <div class='section'>
            <h2>Encoder Only Model with Classification Head</h2>
            <p>
            The model architecture utilizes PyTorch's neural network modules to create a powerful encoder-based classifier.
            </p>
            <details>
                <summary>Read more</summary>
                <ul>
                    <li>nn.Linear(input_size, hidden_size): Embedding layer projects input features to a higher-dimensional space.</li>
                    <li>nn.Embedding(input_length, hidden_size): Adds positional information to the embeddings.</li>
                    <li>nn.Transformer(hidden_size, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=0.5): Captures long-range dependencies in log sequences.</li>
                    <li>nn.Linear(hidden_size, output_size): Classification head maps transformer output to binary classification.</li>
                </ul>
                <p>
                The forward pass processes embeddings with positional encoding through the Transformer, using the final hidden state for classification.
                </p>
            </details>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Model_Selection_02.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Placeholder 5
    st.markdown("""
    <div class='section'>
        <h2>A focus on Precision</h2>
        <p>
        In anomaly detection systems, prioritizing precision over recall leads to more effective and reliable alerts. This approach minimizes false positives, which are a significant source of disruption and resource waste.
        </p>
        <details>
            <summary>Read more</summary>
            <p>
            The benefits of focusing on precision in anomaly detection include:
            <ul>
                <li>Actionable Alerts: Reducing false positives ensures that alerts are more likely to represent true anomalies, making them more actionable for responders.</li>
                <li>Resource Efficiency: Fewer false alarms mean less wasted time and resources investigating non-issues.</li>
                <li>Maintained Trust: High precision helps maintain trust in the alerting system. When an alert is raised, stakeholders can be confident it's likely a true anomaly.</li>
                <li>Reduced Alert Fatigue: By minimizing false positives, the system helps prevent alert fatigue among operators, ensuring they remain responsive to genuine threats.</li>
            </ul>
            </p>
        </details>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/Fundamentals_04.png', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    
    st.markdown("""
        <div class='section'>
            <h2>Upcoming: A Journey of Continuous Improvement</h2>
            <p>
            This marks the beginning of MindMesh's journey to address critical business needs in anomaly detection. Our focus remains on enhancing precision while maintaining high recall, building upon the foundational work with Transformers.
            </p>
            <p>
            Some of the roadmap items under consideration are:
            <ul>
                <li>Improving Model Performance: 
                    <ul>
                        <li>Fine-tuning Transformer hyperparameters</li>
                        <li>Incorporating additional relevant features</li>
                    </ul>
                </li>
                <li>Exploring Complementary Model Architectures:
                    <ul>
                        <li>RNNs (Recurrent Neural Networks): Effective for sequential data processing</li>
                        <li>CNNs (Convolutional Neural Networks): Useful for detecting local patterns in data</li>
                        <li>GRUs (Gated Recurrent Units): Efficient at capturing long-term dependencies</li>
                    </ul>
                </li>
                <li>Practical Implementation:
                    <ul>
                        <li>Developing additional tools for seamless integration</li>
                        <li>Building an Agent Team to manage and respond to alerts effectively</li>
                    </ul>
                </li>
            </ul>
            </p>
        </div>
        <hr>
    """, unsafe_allow_html=True)


