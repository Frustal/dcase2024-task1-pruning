import os
import zipfile
import glob

dataset_dir = os.path.dirname(os.path.abspath(__file__))
zip_files = glob.glob(os.path.join(dataset_dir, '*.zip'))

for zf in zip_files:
    print(f"Extracting {os.path.basename(zf)}...")
    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

print("Done extracting all files!")
