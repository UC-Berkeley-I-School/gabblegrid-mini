import os
from datetime import datetime

def generate_project_structure(base_path, skip_files, skip_folders):
    folder_structure = []
    file_contents = {}
    exclude_dirs = {'97.Archive', '94.Project_Transcript', '.ipynb_checkpoints', '__pycache__'}.union(skip_folders)

    for root, dirs, files in os.walk(base_path):
        # Filter out unwanted directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        relative_path = os.path.relpath(root, base_path)
        if any(relative_path.startswith(folder) for folder in skip_folders):
            continue

        level = relative_path.count(os.sep)
        indent = ' ' * 4 * level
        folder_structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f in skip_files:
                continue
            if f.endswith(('.py', '.ipynb', '.sh')):
                folder_structure.append(f"{subindent}{f}")
                file_path = os.path.join(root, f)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_contents[file_path] = file.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as file:
                            file_contents[file_path] = file.read()
                    except Exception as e:
                        print(f"Skipping file {file_path} due to encoding issues: {e}")
    
    return folder_structure, file_contents

def write_to_file(output_file, folder_structure, file_contents, base_path):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("About\n")
        f.write("-----\n")
        f.write("This is a deep learning and AI Agents project in the field of log anomaly detection. ")
        f.write("The application is built on Python/Streamlit and Section 1 will give you the folder/file structure ")
        f.write("whereas Section 2 will give you the current code in each of the files.\n\n")
        f.write("Please read the entirety of Section 1 and Section 2 and follow instructions at the end.\n\n")
        f.write("When responding to the instructions please answer with the specific section of the code, not the entirety of the whole application or even file. I would like to be as precise as possible and make surgical changes.\n\n")

        f.write("----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")
        f.write("SECTION 1: FOLDER STRUCTURE WITH FILES\n")
        f.write("----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")
        for item in folder_structure:
            f.write(f"{item}\n")
        
        f.write("\n----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")
        f.write("SECTION 2: SECTION, FOLDER, FILE ---> CODE FOR EACH FILE\n")
        f.write("----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")
        for file_path, content in file_contents.items():
            relative_path = os.path.relpath(file_path, base_path)
            f.write(f"\n###################################################################\n")
            f.write(f"############## File: {relative_path} ##############################\n")
            f.write(f"###################################################################\n\n")
            f.write(content)
            f.write("\n\n")

        f.write("\n----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")
        f.write("SECTION 3: INSTRUCTIONS\n")
        f.write("----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = script_dir  # Adjusted to the project directory
    output_dir = os.path.join(script_dir, '94.Project_Transcript')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_file = os.path.join(output_dir, f'{timestamp}.txt')

    # Define files and folders to skip
    skip_files = ['__init__.py', 
                  'transcript_playground_brief.py',
                  'footer.py',
                  'transcript_playground_full.py',
                  'playground_beta_main.py'
                 ]
    skip_folders = ['95.Temp', '96.Archive']  # Add folders you want to skip here

    folder_structure, file_contents = generate_project_structure(base_path, skip_files, skip_folders)
    write_to_file(output_file, folder_structure, file_contents, base_path)

    print(f"Project structure and file contents have been written to {output_file}")