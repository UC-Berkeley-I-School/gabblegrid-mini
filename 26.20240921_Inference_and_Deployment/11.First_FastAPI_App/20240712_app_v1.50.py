import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Load your model and other necessary components
model_folder_path = '/home/ubuntu/efs-w210-capstone-ebs/04.Saved_Models/01.Non_Overlapping/20240711_180_GOLD'
trained_model_file = f"{model_folder_path}/20240711_055019_20240710_(I-4)_Transformers_Non_Overlapping_EarlyStopping_MaxRuns_v1.00_run_133_checkpoint.pt"

# Model parameters
input_length = 20
hidden_size = 64
dropout = 0.3
num_layers = 2
input_size = 55  # Ensure this matches the input size used during training after dropping the first column

# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = nn.Embedding(input_length, hidden_size)
        self.transformer = nn.Transformer(hidden_size, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x

# Load the model
model = TransformerModel(input_size, hidden_size, num_layers, 1, dropout)
model.load_state_dict(torch.load(trained_model_file))
model.eval()

app = FastAPI()

# Define the data model for the prediction request
class PredictionRequest(BaseModel):
    sequence: list

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the prediction API. Use POST /predict to make predictions."}

# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input to tensor
        input_tensor = torch.tensor(request.sequence, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output).item()
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
