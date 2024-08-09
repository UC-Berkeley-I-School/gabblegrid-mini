import streamlit as st

def display_models_tab():
    st.markdown("""
        <style>
            .section p, .section li, .section h2, .section table, .section td, .section th {
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
            <h2>Introduction to Transformers</h2>
            <p>
            Transformers are designed to handle sequential data and rely entirely on attention mechanisms to draw global dependencies between input and output. 
            Unlike traditional RNNs (Recurrent Neural Networks), Transformers do not require the sequential data to be processed in order, which allows for much greater parallelization.
            </p>
            <p>
            Below is a detailed breakdown of the Transformer model and its performance compared to other models.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Transformer Model Architecture</h2>
            <p>
            The Transformer model architecture involves the following key components:
            </p>
            <ul>
                <li><strong>Input Embedding:</strong> Converts the input tokens into dense vectors of fixed size.</li>
                <li><strong>Positional Encoding:</strong> Adds positional information to the input embeddings.</li>
                <li><strong>Multi-Head Attention:</strong> Allows the model to focus on different parts of the input sequence simultaneously.</li>
                <li><strong>Add & Normalize:</strong> Residual connections followed by layer normalization.</li>
                <li><strong>Feed Forward:</strong> Fully connected feed-forward network applied to each position.</li>
                <li><strong>Encoder:</strong> Processes the input sequence to produce a representation.</li>
                <li><strong>Decoder:</strong> Generates the output sequence using the encoder's output.</li>
                <li><strong>Masked Multi-Head Attention:</strong> Prevents the model from attending to future tokens.</li>
                <li><strong>Output Embedding:</strong> Converts the decoder's output into the desired format.</li>
                <li><strong>Softmax:</strong> Converts the final output into probabilities.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Model Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Recall</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Transformer</td>
                        <td>60%</td>
                        <td>The Transformer model achieves the highest recall among the four models tested, indicating it is more effective at identifying anomalies.</td>
                    </tr>
                    <tr>
                        <td>LSTM</td>
                        <td>25%</td>
                        <td>The LSTM model has a high number of true negatives, indicating it is good at identifying normal instances. However, it struggles with correctly identifying anomalies.</td>
                    </tr>
                    <tr>
                        <td>GRU</td>
                        <td>25%</td>
                        <td>The GRU model also has a high number of true negatives but struggles with precision and recall, similar to the LSTM model.</td>
                    </tr>
                    <tr>
                        <td>CNN</td>
                        <td>5%</td>
                        <td>The CNN model has the lowest recall, indicating it is the least effective at identifying anomalies among the tested models.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Transformer Model Parameters</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>max_events</td>
                        <td>500</td>
                        <td>The maximum number of unique events considered for the input feature space.</td>
                    </tr>
                    <tr>
                        <td>input_length</td>
                        <td>10</td>
                        <td>The number of time steps in each input sequence.</td>
                    </tr>
                    <tr>
                        <td>gap</td>
                        <td>1</td>
                        <td>The gap between the end of the input sequence and the start of the prediction period.</td>
                    </tr>
                    <tr>
                        <td>prediction_period</td>
                        <td>1</td>
                        <td>The number of time steps ahead for which the prediction is made.</td>
                    </tr>
                    <tr>
                        <td>test_size</td>
                        <td>0.2</td>
                        <td>The proportion of the dataset to include in the test split.</td>
                    </tr>
                    <tr>
                        <td>shuffle</td>
                        <td>FALSE</td>
                        <td>Indicates whether to shuffle the dataset before splitting.</td>
                    </tr>
                    <tr>
                        <td>hidden_size</td>
                        <td>64</td>
                        <td>The number of features in the hidden state of the Transformer.</td>
                    </tr>
                    <tr>
                        <td>num_layers</td>
                        <td>2</td>
                        <td>The number of encoder and decoder layers in the Transformer.</td>
                    </tr>
                    <tr>
                        <td>num_epochs</td>
                        <td>50</td>
                        <td>The number of training epochs.</td>
                    </tr>
                    <tr>
                        <td>batch_size</td>
                        <td>16</td>
                        <td>The number of samples per gradient update.</td>
                    </tr>
                    <tr>
                        <td>learning_rate</td>
                        <td>0.001</td>
                        <td>The step size at each iteration while moving toward a minimum of a loss function.</td>
                    </tr>
                    <tr>
                        <td>dropout</td>
                        <td>0.3</td>
                        <td>The dropout rate to prevent overfitting.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Model Performance</h2>
            <p>
            The Transformer model achieves the highest recall (60%) among the models tested, indicating its effectiveness in identifying anomalies. 
            However, it still struggles with false positives, and improving precision while maintaining high recall is a key focus moving forward.
            </p>
            <p>
            The LSTM and GRU models both have recall rates of 25%, indicating a need for improvement in identifying anomalies accurately. 
            The CNN model has the lowest recall at 5%, making it the least effective among the tested models.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <h2>Future Enhancements</h2>
            <p>
            Future enhancements will focus on improving the precision of the Transformer model while maintaining high recall. 
            This involves tuning hyperparameters, adding more features, and exploring different model architectures and techniques.
            </p>
        </div>
    """, unsafe_allow_html=True)