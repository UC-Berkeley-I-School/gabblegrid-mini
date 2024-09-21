import os
import shutil
from datetime import datetime

def backup_project(project_folder, target_folder):
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Create the backup folder name
    backup_folder_name = f"{timestamp}"
    
    # Define the full path for the backup folder
    backup_folder_path = os.path.join(target_folder, backup_folder_name)
    
    # Create the backup folder
    os.makedirs(backup_folder_path, exist_ok=True)
    
    # List of folders to exclude
    exclude_folders = {'97.Archive', '94.Project_Transcript'}
    
    # Copy the entire project folder to the backup folder, excluding specified folders
    for item in os.listdir(project_folder):
        if item not in exclude_folders:
            s = os.path.join(project_folder, item)
            d = os.path.join(backup_folder_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
    
    print(f"Backup completed successfully. Backup folder: {backup_folder_path}")

if __name__ == "__main__":
    project_folder = "/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/project"
    target_folder = "/home/ubuntu/efs-w210-capstone-ebs/06.Inference_and_Deployment/00.Git_Code/00.Full_Project_Backups"
    
    backup_project(project_folder, target_folder)