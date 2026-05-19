# SIMPLE TAR CREATION SCRIPT

import os
import tarfile

# CONFIG
DATASET_NAME = "Dataset_name"
IMAGE_DIR = r"images_folder_route"
OUTPUT_DIR = r"tar_folder_route"


# CREATE OUTPUT DIR
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# IMAGE EXTENSIONS
SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

# FIND ALL IMAGES
all_files = []

for file_name in os.listdir(IMAGE_DIR):

    if not file_name.lower().endswith(
        SUPPORTED_EXTENSIONS
    ):
        continue

    full_path = os.path.join(
        IMAGE_DIR,
        file_name
    )

    all_files.append(full_path)

all_files = sorted(all_files)

print("\nTOTAL IMAGES:")
print(len(all_files))

# CREATE TAR
tar_path = os.path.join(
    OUTPUT_DIR,
    f"{DATASET_NAME}.tar"
)

print("\nCREATING:")
print(tar_path)

with tarfile.open(
    tar_path,
    "w"
) as tar:

    for file_path in all_files:

        file_name = os.path.basename(
            file_path
        )

        arcname = os.path.join(
            DATASET_NAME,
            file_name
        )

        tar.add(
            file_path,
            arcname=arcname
        )

print("\nDONE")
