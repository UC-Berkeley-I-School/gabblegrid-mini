import os
from datetime import datetime

def generate_project_structure(base_path, include_dirs, include_files):
    folder_structure = []
    file_contents = {}
    exclude_dirs = {'__pycache__', '.ipynb_checkpoints', '.cache'}

    for root, dirs, files in os.walk(base_path):
        # Filter out unwanted directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        folder_structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if any(d in root for d in include_dirs) or f in include_files:
                folder_structure.append(f"{subindent}{f}")
                file_path = os.path.join(root, f)
                if f.endswith(('.py', '.ipynb', '.sh')):
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

    include_dirs = {'config', 'content', 'files', 'group', 'model', 'utils'}
    include_files = {'main.py'}

    folder_structure, file_contents = generate_project_structure(base_path, include_dirs, include_files)
    write_to_file(output_file, folder_structure, file_contents, base_path)

    print(f"Project structure and file contents have been written to {output_file}")
