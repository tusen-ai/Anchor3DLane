# import tarfile
# import os

# # The directory containing your .tar files
# source_dir = '/home/ec2-user/annas_workstation/Anchor3DLane-main/images_openlane'

# # The directory where you want to extract your files
# extract_to_dir = '/home/ec2-user/annas_workstation/Anchor3DLane-main/data/OpenLane/images'

# # Loop through all files in the source directory
# for filename in os.listdir(source_dir):
#     if filename.endswith('.tar') or filename.endswith('.tar.gz') or filename.endswith('.tar.bz2'):
#         # Construct the full path to the file
#         file_path = os.path.join(source_dir, filename)
#         # Open the .tar file
#         with tarfile.open(file_path) as file:
#             # Extract all the contents into the directory specified
#             file.extractall(path=extract_to_dir)
#         print(f'Extracted {filename}')

# import os
# import shutil

# # Path to the directory containing the 16 folders
# source_dir = '/home/ec2-user/annas_workstation/Anchor3DLane-main/valdata'

# # Path to the directory where you want to move all the subfolders
# target_dir = '/home/ec2-user/annas_workstation/Anchor3DLane-main/all_valdata'

# # Ensure the target directory exists
# os.makedirs(target_dir, exist_ok=True)

# # Loop through all subdirectories in the source directory
# for folder_name in os.listdir(source_dir):
#     folder_path = os.path.join(source_dir, folder_name)
    
#     # Check if it's a directory
#     if os.path.isdir(folder_path):
#         # Loop through all subdirectories in this folder
#         for subfolder_name in os.listdir(folder_path):
#             subfolder_path = os.path.join(folder_path, subfolder_name)
            
#             # Construct the target path for the subdirectory
#             target_subfolder_path = os.path.join(target_dir, subfolder_name)
            
#             # Move the subdirectory to the target directory
#             # Check if the target subdirectory already exists to avoid overwriting
#             if not os.path.exists(target_subfolder_path):
#                 shutil.move(subfolder_path, target_dir)
#                 print(f'Moved {subfolder_path} to {target_dir}')
#             else:
#                 print(f'Target subdirectory {target_subfolder_path} already exists, consider renaming or manually handling.')

import json
import tqdm

max_lanes = 0
line_with_max_lanes = None

with open('data/zod_dataset/data_splits/validation.json', 'r') as file:
    for line_number, line in tqdm.tqdm(enumerate(file, 1)):
        try:
            data = json.loads(line)
            lane_lines = data.get("lane_lines", [])
            num_lanes = len(lane_lines)
            if num_lanes > max_lanes:
                max_lanes = num_lanes
                line_with_max_lanes = data
        except json.JSONDecodeError:
            print(f"Skipping line {line_number} due to JSON decoding error.")

print(f"The maximum number of lane lines is: {max_lanes}")