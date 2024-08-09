# playground_text.py

playground_intro = """
<div class='section'>
    <p>Welcome to the Playground! This is where the magic happens. Here, you can test the anomaly detection process and see our AI agents in action. 
    The GabbleGrid solution leverages a powerful Transformer model designed specifically for anomaly detection in cloud services.</p>
</div>
"""

playground_intro_expanded = """
<div class='section'>
    <p>The playground takes you through the anomaly detection process, allowing you to see model performance and agents in action. Explore the Playground 
    and witness how our AI agents keep the cloud running smoothly!</p>
</div>
"""

key_parameters = """
<div class='section'>
    <h5 style="color: grey;">Parameters: Explained</h5>
    <p>Note: Shown above is a sample depiction of a 6 period inference set, with 4 for Observation, a Gap period of 1 followed by a Prediction period of 1</p>
    <ul>
        <li><strong>Start Date:</strong> This is the beginning of the time period from which you would like to test. Select the date here. Note: This list is limited to the two months in 2005 of the Blue/Gele-L dataset used for inference</li>
        <li><strong>Start Time:</strong> Continuing on from the Start Date, the start time is in 5 minute increments for more specificity.</li>
        <li><strong>Sliding Window:</strong> By default, only 'Sequential' windows are available to select currently. This implies that each sample period (which includes the observation, gap and prediction period) is independent of the next and represents a IID condition. The sample periods are also sequential in time</li>
        <li><strong>AutoPicker:</strong> By default, only the 'Manual, option is available currently. This represents that the selection of the transformer model will be rule based and does not involve AI Agents. In a upcoming release, we will release the capability of Agent selected models</li>
        <li><strong>Max Events:</strong> Number of principal components to keep after applying PCA to the EventID columns. In effect, for each 5 minute time period, we distill on average, 35000 events downn to the Max Events value selected here. This represents a reduction factor of over 99.9%</li>
        <li><strong>Number of Samples:</strong> This is the number of sequential tests to trigger for inference. For example, say each sample is defined by say 10 periods ---say 8 observation, 1 gap and 1 prediction; this works out to 50 minutes as each period is set to a standard of 5 minutes. So selecting a value of 20 samples means that inference will be triggered for a sequence of 20 consequitive 10 period samples. In essence, the total elapsed time from the start of inference to the end will be 20 X 50 minutes = 1,000 minutes</li>
        <li><strong>Observation Period:</strong> Represents the number of 5 minute periods to observe before making a prediction. Setting a value of 8 implies that the model will observe for 40 minutes (8 X 5) before making a prediction</li>
        <li><strong>Gap Period:</strong> Number of 5-minute intervals after observation before the model is expected to make a prediction. In the above example, say we had a observation period of 8 periods (or 40 minutes), then a gap of 1 period represent a 5 minute period of unseen data which providies a separation between the observation and the prediction.</li>
        <li><strong>prediction_period:</strong> This represents the 5-minute period for which the prediction is being made. Following the earlier example, a value of 1 here implies that the model is going to predict the outcome of the 10th period (following 8 observation + 1 gap periods</li>
    </ul>
</div>
"""

