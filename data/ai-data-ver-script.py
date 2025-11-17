import os
from pathlib import Path
from collections import Counter

# check file count in a folder and its subdirs, seperate into file types
def count_files_in_folder(folder_path):
    total_files = 0
    file_types = Counter()
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
        for file in files:
            ext = Path(file).suffix
            file_types[ext] += 1
    print(f'File types in {folder_path}:')
    for ext, count in file_types.items():
        print(f'  {ext}: {count}')
    return total_files

pklot_folder = 'CNR-EXT_FULL_IMAGE_1000x750'
file_count = count_files_in_folder(pklot_folder)
print(f'Total files in {pklot_folder}: {file_count}')

