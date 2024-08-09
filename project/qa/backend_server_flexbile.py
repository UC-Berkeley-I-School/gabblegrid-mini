import sys
import os
from flask import Flask, request, jsonify
import numpy as np
import torch
# Add the project directory to the system path
sys.path.append('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project')
from playground_beta.beta_model.load_model import load_model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_test = np.array(data['X_test'])
    input_length = data['input_length']
    gap = data['gap']
    prediction_period = data['prediction_period']
    max_events = data['max_events']
    
    print(f"Received parameters - input_length: {input_length}, gap: {gap}, prediction_period: {prediction_period}, max_events: {max_events}")
    print(f"X_test shape before processing: {X_test.shape}")
    
    # Load the model
    csv_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv'
    model, model_info = load_model(input_length, gap, prediction_period, max_events, csv_path)
    
    # Extract just the filename from the model_info dictionary
    if isinstance(model_info, dict) and 'trained_model_file' in model_info:
        model_name = os.path.basename(model_info['trained_model_file'])
    else:
        model_name = 'unknown_model'
    
    print(f"Selected model file: {model_name}")
    
    print(f"X_test shape after processing: {X_test.shape}")
    
    # Convert X_test to tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    print(f"X_test_tensor shape: {X_test_tensor.shape}")
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_outputs_sigmoid = torch.sigmoid(test_outputs)
        predictions = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).flatten()
    
    response = {'predictions': predictions.tolist(), 'model_name': model_name}
    print(f"Returning response: {response}")
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


############# OLDER CODE REFERENCE ONLY ##########################

# import sys
# from flask import Flask, request, jsonify
# import numpy as np
# import torch

# # Add the project directory to the system path
# sys.path.append('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project')

# from playground_beta.beta_model.load_model import load_model

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     X_test = np.array(data['X_test'])
#     input_length = data['input_length']
#     gap = data['gap']
#     prediction_period = data['prediction_period']
#     max_events = data['max_events']
    
#     print(f"Received parameters - input_length: {input_length}, gap: {gap}, prediction_period: {prediction_period}, max_events: {max_events}")
#     print(f"X_test shape before processing: {X_test.shape}")
    
#     # Load the model
#     csv_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv'
#     model, _ = load_model(input_length, gap, prediction_period, max_events, csv_path)
    
#     # No need to drop the first column, ensuring input size matches model's expectations
#     print(f"X_test shape after processing: {X_test.shape}")
    
#     # Convert X_test to tensor
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     print(f"X_test_tensor shape: {X_test_tensor.shape}")

#     with torch.no_grad():
#         test_outputs = model(X_test_tensor)
#         test_outputs_sigmoid = torch.sigmoid(test_outputs)
#         predictions = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).flatten()
    
#     return jsonify({'predictions': predictions.tolist()})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


########################################################################################################
########################### Changes to extract the model name ##########################################
########################################################################################################

# import sys
# import os
# from flask import Flask, request, jsonify
# import numpy as np
# import torch

# # Add the project directory to the system path
# sys.path.append('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project')

# from playground_beta.beta_model.load_model import load_model

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     X_test = np.array(data['X_test'])
#     input_length = data['input_length']
#     gap = data['gap']
#     prediction_period = data['prediction_period']
#     max_events = data['max_events']
    
#     print(f"Received parameters - input_length: {input_length}, gap: {gap}, prediction_period: {prediction_period}, max_events: {max_events}")
#     print(f"X_test shape before processing: {X_test.shape}")
    
#     # Load the model
#     csv_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv'
#     model, _ = load_model(input_length, gap, prediction_period, max_events, csv_path)
    
#     # Get the model name from the file path
#     model_dir = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/20240713_No_Overlap_180'
#     # print(f"Checking for model files in directory: {model_dir}")

#     model_files = os.listdir(model_dir)
#     # print(f"Files found in model directory: {model_files}")
    
#     model_name = model_files[0] if model_files else "unknown_model"
#     print(f"Selected model name: {model_name}")
    
#     # No need to drop the first column, ensuring input size matches model's expectations
#     print(f"X_test shape after processing: {X_test.shape}")
    
#     # Convert X_test to tensor
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     print(f"X_test_tensor shape: {X_test_tensor.shape}")

#     with torch.no_grad():
#         test_outputs = model(X_test_tensor)
#         test_outputs_sigmoid = torch.sigmoid(test_outputs)
#         predictions = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).flatten()
    
#     response = {'predictions': predictions.tolist(), 'model_name': model_name}
#     print(f"Returning response: {response}")
    
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

########################################################################################################
########################### Changes to extract the model name ##########################################
########################################################################################################

# import sys
# import os
# from flask import Flask, request, jsonify
# import numpy as np
# import torch

# # Add the project directory to the system path
# sys.path.append('/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project')

# from playground_beta.beta_model.load_model import load_model

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     X_test = np.array(data['X_test'])
#     input_length = data['input_length']
#     gap = data['gap']
#     prediction_period = data['prediction_period']
#     max_events = data['max_events']
    
#     print(f"Received parameters - input_length: {input_length}, gap: {gap}, prediction_period: {prediction_period}, max_events: {max_events}")
#     print(f"X_test shape before processing: {X_test.shape}")
    
#     # Load the model
#     csv_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files/00.Tracker/20240713_Transformers_NonOverlapping_180.csv'
#     model, model_name = load_model(input_length, gap, prediction_period, max_events, csv_path)
    
#     print(f"Selected model file: {model_name}")
    
#     print(f"X_test shape after processing: {X_test.shape}")
    
#     # Convert X_test to tensor
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     print(f"X_test_tensor shape: {X_test_tensor.shape}")

#     with torch.no_grad():
#         test_outputs = model(X_test_tensor)
#         test_outputs_sigmoid = torch.sigmoid(test_outputs)
#         predictions = (test_outputs_sigmoid.cpu().numpy() > 0.5).astype(int).flatten()
    
#     response = {'predictions': predictions.tolist(), 'model_name': model_name}
#     print(f"Returning response: {response}")
    
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

