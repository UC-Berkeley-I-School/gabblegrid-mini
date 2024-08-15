import os
from datetime import datetime

def generate_project_structure(base_path, skip_files, exclude_dirs):
    folder_structure = []
    file_contents = {}

    for root, dirs, files in os.walk(base_path):
        # Filter out unwanted directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        relative_path = os.path.relpath(root, base_path)
        if any(folder in relative_path.split(os.sep) for folder in exclude_dirs):
            continue

        level = relative_path.count(os.sep)
        indent = '\t' * level
        folder_structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = '\t' * (level + 1)
        for f in files:
            if f in skip_files:
                continue
            if f.endswith(('.py', '.sh', '.yaml', '.txt')):
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

        f.write("To recap, the application is built on Python/Streamlit and Section 1 has the folder/file structure ")
        f.write("whereas Section 2 has the current code in each of the files.\n\n")
        f.write("When responding to the instructions please answer with the specific section of the code, not the entirety of the whole application or even file. I would like to be as precise as possible and make surgical changes.\n\n")
        f.write("Please do not regurgitate the code if it has not changed.\n\n")
        f.write("Please DO NOT print/display the entire code unless asked for.\n\n")
        f.write("Please always give me the before and after for the code change you propose.\n\n")
        f.write("\n----------------------------------------------------------------\n")
        f.write("Here are these key files\n\n")
        # f.write("\n----------------------------------------------------------------\n")
        # f.write("03.20240715_143154_orig_input_w_seq_info_FINAL.parquet.\n\n")
        # f.write("Location: /home/ubuntu/efs-w210-capstone-ebs/04A.Local_Data_Files.\n\n")
        # f.write("\n----------------------------------------------------------------\n")
        # f.write("03B.20240716_072206_orig_parquet_mapper_agents.parquet.\n\n")
        # f.write("Location: /home/ubuntu/efs-w210-capstone-ebs/04A.Local_Data_Files.\n\n")
        # f.write("\n----------------------------------------------------------------\n")
        # f.write("08.20240716031626_event_ID_int_template_mapping.csv.\n\n")
        # f.write("Location: /home/ubuntu/efs-w210-capstone-ebs/04A.Local_Data_Files.\n\n")
        # f.write("\n----------------------------------------------------------------\n")
        # f.write("03B.{timestamp}_agent3_non_overlap_model2_consl.csv.\n\n")
        # f.write("Location: /home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/05.Local_Results_Tracker\n\n")

        f.write("\n----------------------------------------------------------------\n")
        f.write("REF_ONLY_Group3_Agent_A_Historical_Weather_Retriever_Notebook.txt\n\n")
        f.write("This is the notebook version of the Agent A in Group 3 whose task is the run inference and get the detailed results")
        f.write("\n----------------------------------------------------------------\n")
        f.write("REF_ONLY_Group3_Agent_B_Historical_Weather_Plotter_Notebook.txt\n\n")
        f.write("This is the notebook version of the Agent B in Group 3 whose task is to plot the results from Agent A")

        f.write("\n----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")
        f.write("SECTION 4: TASKS\n")
        f.write("----------------------------------------------------------------\n")
        f.write("----------------------------------------------------------------\n")


        f.write("Once you have read through the contents of this message, please re-read the code\n\n")
        f.write("in these two source files - Consider them as the framework on which we will develop now\n\n")
        f.write("File1: playground_main.py\n\n")
        f.write("File2: playground_weather_inference.py\n\n")
        f.write("Once you have read both these reference files, you will re-read these two target files again: \n\n")
        f.write("File1: REF_ONLY_Group3_Agent_A_Historical_Weather_Retriever_Notebook.txt\n\n")
        f.write("File2: REF_ONLY_Group3_Agent_B_Historical_Weather_Plotter_Notebook.txt\n\n")
        f.write("Both these target files is working code from a jupyter notebook\n\n")
        f.write("The task is to translate the target code into two agents\n\n")
        f.write("First Agent: Historical Weather Data Retriever \n\n")
        f.write("Second Agent: Historical Weather Data Plotter\n\n")
        f.write("For this I have created a new file 'playground_historical_weather_inference.py'\n\n")
        f.write("which will be similar to the existing file File2: playground_weather_inference.py\n\n")
        f.write("All the code you will give me now will go into playground_historical_weather_inference.py\n\n")
        f.write("To make this work in the reference agent format, you will need to recast the target code\n\n")
        f.write("Let us start by first generating the content for Historical Weather Retriever  \n\n")
        f.write("Once complete, we will move on to the Historical Weather Data Plotter Agent code\n\n")
        f.write("And Remember, please always give me targetted/specific code and not the full file\n\n")
        f.write("I will expllicitely ask if I need code for the entire file\n\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = script_dir  # Adjusted to the project directory
    output_dir = os.path.join(script_dir, '94.Project_Transcript')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_file = os.path.join(output_dir, f'{timestamp}.txt')

    # Define files to skip and directories to exclude
    skip_files = [ 
                '__init__.py'
                , 'transcript_playground_brief.py'
                , 'footer.py'
                , 'transcript_playground_full.py'
                , 'playground_main.py'
                , 'generate_project_transcript.py'
                , 'transcript_playground_full.py'
                , 'backup_project_full.py'
                , 'home_tab_backup.py'
                , 'about_us_tab.py'
                , 'design_tab.py'
                # , 'playground_inference.py'
                , 'playground_text.py'
                , 'why_agents_tab.py'
                , 'tech_tab.py'
                , 'models_tab.py'
                , 'terms_of_service.py'
                , 'privacy_policy.py'
                , 'plug_n_play_tab.py'
                , 'plotting.py'
                , 'parameter_sourcing.py'
                , 'autogen_setup.py'
                , 'data_processing.py'
                 # , 'inference.py'
                , 'REF_ONLY_nginx_sites-available_default.txt'
                , 'REF_ONLY_nginx_conf.txt'
                , 'REF_ONLY_gabblegrid_service.txt'
                , 'main-backup.py'
                , 'config.yaml'
                , 'REF_ONLY_03B.20240716_072206_orig_parquet_mapper_agents_parquet.txt'
                , 'REF_ONLY_03.20240715_143154_orig_input_w_seq_info_FINAL_parquet.txt'
                , 'REF_ONLY_AGENT3_CODE.txt'
                , 'main_backup.py'
                , 'admin_tab.py'
                # , 'playground_config.py'
                # , 'playground_utils.py'
                , 'REF_ONLY_Group3_Agent_A_Historical_Weather_Retriever_Notebook.txt'
                , 'REF_ONLY_Group3_Agent_B_Historical_Weather_Plotter_Notebook.txt'
                , 'playground_weather_inference.py'
                # , 'playground_ui.py'
                # , 'main.py'
                , 'playground_config.py'
                , 'home_tab.py'
                # , 'playground_log_inference.py'
                , 'adhoc.txt'
                 ]
    exclude_dirs = {'97.Archive'
                , '94.Project_Transcript'
                , '.ipynb_checkpoints'
                , '__pycache__'
                , '00.Full_Project_Backups'
                , '03.Local_Inference_Eval_Files'
                , 'feature_dev'
                , '04.Local_Other_Files'
                , '02.Local_Data_Files'
                , '01.Local_Model_Files'
                # , 'playground'
                , '96.Originals'
                , 'files'
                , '.cache'
                , 'group'
                # , 'utils'
                , 'model'
                , 'models_tab.py'
                , 'tech_tab.py'
                , 'plug_n_play'
                , '00.Key_REF_Notebooks'
                , 'config'
                , 'qa'
                , 'admin'
                , '00.KEY_ORIGINALS'
                   }

    folder_structure, file_contents = generate_project_structure(base_path, skip_files, exclude_dirs)
    write_to_file(output_file, folder_structure, file_contents, base_path)

    print(f"Project structure and file contents have been written to {output_file}")