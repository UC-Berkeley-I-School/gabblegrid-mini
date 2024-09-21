import os
import datetime
from datetime import timezone
import gzip

# Define paths
data_dir = '/home/sagemaker-user/11.Data/01.BGL/01.Raw_CFDR'
output_dir = '/home/sagemaker-user/11.Data/01.BGL/03.Parsed_CFDR'
bgl_file = 'bgl2.gz'
parsed_file = 'parsed.csv'
template_file = '/home/sagemaker-user/08.GIT_Repos_REF/02.ait-aecid/templates/BGL_templates.csv'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the parser function
def parse_bgl_logs(data_dir, output_dir, bgl_file, parsed_file, template_file):
    # Set file paths
    bgl_file_path = os.path.join(data_dir, bgl_file)
    output_file_path = os.path.join(output_dir, parsed_file)
    
    templates = []
    anomalous_sequences = set()
    
    print('Get labels ...')
    with gzip.open(bgl_file_path, 'rt') as log_file:
        for line in log_file:
            line = line.strip('\n ')
            line_parts = line.split(' ')
            label = line_parts[0]
            seq_id = line_parts[3]
            if label != '-':
                anomalous_sequences.add(seq_id)
    
    events_allow_spaces = [82, 84, 172, 194, 293, 328, 362, 371, 397]  # Line numbers in template file where <*> can represent multiple tokens separated by spaces.

    print('Read lines ...')
    with gzip.open(bgl_file_path, 'rt') as log_file, open(template_file) as templates_file, open(output_file_path, 'w+') as ext_file:
        header = 'id;event_type;seq_id;time;label;eventlabel'
        ext_file.write(header + '\n')
        for line in templates_file:
            template = line.strip('\n').rstrip(' ').split('<*>')
            templates.append(template)
        cnt = 0
        for line in log_file:
            cnt += 1
            if cnt % 50000 == 0:
                print(str(cnt) + ' lines processed')
            line = line.strip('\n ')
            template_id = None
            line_parts = line.split(' ')
            seq_id = line_parts[3]
            eventlabel = line_parts[0]
            if eventlabel == '-':
                eventlabel = 'Normal'
            if seq_id in anomalous_sequences:
                label = "Anomaly"
            else:
                label = "Normal"
            found_params = []
            preamble = line_parts[:9]
            line = ' '.join(line_parts[9:])
            for i, template in enumerate(templates):
                if i == 382:  # E384: Line consists of just a number
                    if line.isdigit():
                        template_id = i + 1
                        found_params = [line]
                        break
                    else:
                        continue
                if i == 383:  # E385: Line is empty
                    if line == '':
                        template_id = i + 1
                        found_params = ['']
                        break
                    else:
                        continue
                matches = []
                params = []
                cur = 0
                starts_with_wildcard = False
                for template_part in template:
                    if template_part == '' and cur == 0:
                        starts_with_wildcard = True
                    pos = line.find(template_part, cur)
                    if pos == -1 or (' ' in line[cur:pos] and i not in events_allow_spaces) or (not starts_with_wildcard and cur == 0 and pos != 0) or (i == 0 and not line.split(' ')[-1].split(':')[0].isdigit()):
                        matches = []  # Reset matches so that it counts as no match at all
                        break
                    matches.append(pos)
                    if line[cur:pos] != '':
                        params.append(line[cur:pos])
                    cur = pos + len(template_part)
                if len(matches) > 0 and sorted(matches) == matches and (' ' not in line[cur:] or i in events_allow_spaces) and (line[cur:] == '' or template_part == ''):
                    if template_id is not None:
                        print('WARNING: Templates ' + str(template_id) + ' and ' + str(i + 1) + ' both match line ' + line)
                    template_id = i + 1  # offset by 1 so that ID matches with line in template file
                    if line[cur:] != '':
                        params.append(line[cur:])
                    found_params = params  # Store params found for matching template since params variable will be reset when checking next template
                    break
            if template_id is None:
                print('WARNING: No template matches ' + str(line))
            timestamp = datetime.datetime.strptime(line_parts[4], '%Y-%m-%d-%H.%M.%S.%f').replace(tzinfo=timezone.utc).timestamp()
            csv_line = str(cnt) + ';' + str(template_id) + ';' + str(seq_id) + ';' + str(timestamp) + ';' + str(label) + ';' + str(eventlabel)
            ext_file.write(csv_line + '\n')

# Parse the logs
parse_bgl_logs(data_dir, output_dir, bgl_file, parsed_file, template_file)