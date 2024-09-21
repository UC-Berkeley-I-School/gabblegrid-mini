import os
import shutil
from datetime import datetime

def backup_project(project_folder, additional_folders, target_folder):
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
    
    # Create a subfolder for the project content
    project_backup_path = os.path.join(backup_folder_path, 'project')
    os.makedirs(project_backup_path, exist_ok=True)
    
    # Copy the entire project folder to the project subfolder, excluding specified folders
    for item in os.listdir(project_folder):
        if item not in exclude_folders:
            s = os.path.join(project_folder, item)
            d = os.path.join(project_backup_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
    
    # Copy the additional folders to the backup folder
    for folder in additional_folders:
        folder_name = os.path.basename(folder)
        d = os.path.join(backup_folder_path, folder_name)
        if os.path.isdir(folder):
            shutil.copytree(folder, d)
        else:
            shutil.copy2(folder, d)
    
    print(f"Backup completed successfully. Backup folder: {backup_folder_path}")

if __name__ == "__main__":
    project_folder = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project"
    additional_folders = [
        # "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/01.Local_Model_Files",
        # "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/02.Local_Data_Files",
        # "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/03.Local_Inference_Eval_Files"
    ]
    target_folder = "/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/00.Full_Project_Backups"
    
    backup_project(project_folder, additional_folders, target_folder)
