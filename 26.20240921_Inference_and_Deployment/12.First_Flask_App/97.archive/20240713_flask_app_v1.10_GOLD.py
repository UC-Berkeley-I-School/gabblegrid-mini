import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

# Parameters
data_dir = '/home/ubuntu/efs-w210-capstone-ebs/04C.Local_Inference_Eval_Files'
model_folder_path = '/home/ubuntu/efs-w210-capstone-ebs/04B.Local_Model_Files/20240713_Non_Overlapping_Consl_180_FINAL'
trained_model_file = f"{model_folder_path}/20240712_Transformers_Non_Overlapping_run_133_of_180.pt"

# Model parameters
input_length = 20
hidden_size = 64
dropout = 0.3
num_layers = 2

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

# Load the trained model
model = TransformerModel(input_size=55, hidden_size=64, num_layers=2, output_size=1, dropout=0.3)  # Updated input_size to 55
model.load_state_dict(torch.load(trained_model_file))
model.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_test = np.array(data['X_test'])
    X_test = X_test[:, :, 1:]  # Drop the first column

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_outputs_sigmoid = torch.sigmoid(test_outputs)
        test_preds = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).tolist()
    
    return jsonify(predictions=test_preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
