# playground_text.py

playground_intro = """
<div class='section'>
    <p>Welcome to the Playground! This is where the magic happens. Here, you can test the anomaly detection process and see our AI agents in action. 
    The GabbleGrid solution leverages a powerful Transformer model designed specifically for anomaly detection in cloud services. The default settings 
    for our model are:
    </p>
    <ul>
        <li><code>input_size=55</code></li>
        <li><code>hidden_size=64</code></li>
        <li><code>num_layers=2</code></li>
        <li><code>output_size=1</code></li>
        <li><code>input_length=30</code></li>
        <li><code>dropout=0.3</code></li>
    </ul>
    <p>The playground takes you through the anomaly detection process, allowing you to see model performance and agents in action. Explore the Playground 
    and witness how our AI agents keep the cloud running smoothly!</p>
</div>
"""

key_parameters = """
<div class='section'>
    <h2>Key Parameters and Hyperparameters</h2>
    <ul>
        <li><strong>max_events:</strong> Number of principal components to keep after applying PCA to the EventID columns.</li>
        <li><strong>input_length:</strong> Length of the input sequence for the model. Determines how many time steps are included in each sequence used for predictions.</li>
        <li><strong>hidden_size:</strong> Number of features in the hidden state of the Transformer model. Controls the capacity of the model to capture patterns in the data.</li>
        <li><strong>num_layers:</strong> Number of layers in the Transformer model. Affects the depth of the model and its ability to learn complex patterns.</li>
        <li><strong>dropout:</strong> Dropout rate used in the Transformer model. Helps prevent overfitting by randomly dropping units during training.</li>
        <li><strong>gap:</strong> Number of time steps between the end of the input sequence and the prediction period. Ensures non-overlapping sequences for more diverse training data.</li>
        <li><strong>prediction_period:</strong> Number of time steps to predict ahead. Determines how far into the future the model makes predictions.</li>
    </ul>
</div>
"""
