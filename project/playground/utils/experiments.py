from datetime import datetime

def create_timestamp():
    return datetime.now().strftime('%Y%m%d%H%M%S')

def create_experiment_id():
    # Generate the experiment ID based on the current timestamp
    return create_timestamp()
