import torch
from .transformer import TransformerModel

def load_model():
    model_folder_path = '/home/ubuntu/efs-w210-capstone-ebs/04B.Local_Model_Files/20240713_Non_Overlapping_Consl_180_FINAL'
    trained_model_file = f"{model_folder_path}/20240712_Transformers_Non_Overlapping_run_143_of_180.pt"
    model = TransformerModel(input_size=55, hidden_size=64, num_layers=2, output_size=1, input_length=30, dropout=0.3)
    model.load_state_dict(torch.load(trained_model_file))
    model.eval()
    return model
