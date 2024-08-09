import torch
import pandas as pd
from .transformer import TransformerModel

def load_model(input_length, gap, prediction_period, max_events, csv_path, hidden_size=64, dropout=0.3, num_layers=2):
    # Define the model folder path at the start of the function
    model_folder_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/20240713_No_Overlap_180'
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Find the row that matches the given parameters
    match_row = df[(df['Input Length'] == input_length) & 
                   (df['Gap'] == gap) & 
                   (df['Prediction Period'] == prediction_period) & 
                   (df['Max Events'] == max_events) &
                   (df['Hidden Size'] == hidden_size) &
                   (df['Dropout'] == dropout) &
                   (df['Num Layers'] == num_layers)]
    
    if match_row.empty:
        raise ValueError(f"No matching model found for Input Length: {input_length}, Gap: {gap}, Prediction Period: {prediction_period}, Max Events: {max_events}, Hidden Size: {hidden_size}, Dropout: {dropout}, Num Layers: {num_layers}")
    
    # Select the first matching row
    first_match = match_row.iloc[0]
    input_size = int(max_events) + 15  # Ensure input_size is an integer
    
    # Construct the trained model file path using the model_folder_path
    run_number = int(first_match['Run'])
    trained_model_file = f"{model_folder_path}/20240712_Transformers_Non_Overlapping_run_{run_number}_of_180.pt"
    
    # Instantiate the model
    model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, input_length=input_length, dropout=dropout)
    
    # Load the trained model state
    model.load_state_dict(torch.load(trained_model_file))
    model.eval()

    # Log the selected model and parameters
    print(f"Selected model: {trained_model_file}")
    print(f"Model parameters: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, input_length={input_length}, dropout={dropout}")
    
    return model, {
        "trained_model_file": trained_model_file,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "input_length": input_length,
        "dropout": dropout
    }