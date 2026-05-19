# FULL IMAGE FLATTENING AND RENAMING + CSV IMAGE COLUMN REWRITE

import os
import shutil
import pandas as pd

# CONFIG
DATASET_ROOT = r"dataset_images_folder_path"
INPUT_CSV = os.path.join(
    DATASET_ROOT,
    "dataset_csv_name"
)

OUTPUT_ROOT = r"dataset_output_folder_path"
OUTPUT_IMAGE_DIR = os.path.join(
    OUTPUT_ROOT,
    "images"
)
OUTPUT_CSV = os.path.join(
    OUTPUT_ROOT,
    "csv_output_name"
)

# CREATE OUTPUT FOLDERS
os.makedirs(
    OUTPUT_IMAGE_DIR,
    exist_ok=True
)

# LOAD CSV
df = pd.read_csv(INPUT_CSV)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

print("\nCSV COLUMNS:")
print(df.columns.tolist())
print("\nORIGINAL ROWS:", len(df))

# IMAGE EXTENSIONS
SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

# BUILD IMAGE MAPPING FLATENING IMAGE PATH AND APPENDING TO START OF EXISTING NAME
image_mapping = {}

# FIND ALL IMAGES
print("\nSCANNING IMAGES...")

for root, dirs, files in os.walk(DATASET_ROOT):

    for file_name in files:

        if not file_name.lower().endswith(
            SUPPORTED_EXTENSIONS
        ):
            continue

        # ORIGINAL FULL PATH
        old_full_path = os.path.join(
            root,
            file_name
        )

        # RELATIVE PATH
        relative_path = os.path.relpath(
            old_full_path,
            DATASET_ROOT
        )

        # CREATE SAFE FLAT NAME
        new_file_name = relative_path.replace(
            "/",
            "_"
        )

        new_file_name = new_file_name.replace(
            "\\",
            "_"
        )

        # REMOVE DOUBLE UNDERSCORES
        while "__" in new_file_name:

            new_file_name = new_file_name.replace(
                "__",
                "_"
            )

        # NEW FILE PATH
        new_full_path = os.path.join(
            OUTPUT_IMAGE_DIR,
            new_file_name
        )

        # COPY IMAGE
        shutil.copy2(
            old_full_path,
            new_full_path
        )

        # SAVE MAPPING
        image_mapping[
            relative_path
        ] = new_file_name

print("\nTOTAL IMAGES FOUND:")
print(len(image_mapping))

# REWRITE CSV
valid_rows = []
missing_images = []
print("\nREWRITING CSV...")

for idx, row in df.iterrows():

    try:

        # ORIGINAL CSV IMAGE PATH
        original_image_path = str(
            row["image"]
        ).strip()

        original_image_path = original_image_path.replace(
            "\\",
            "/"
        )

        # VERIFY EXISTS IN MAPPING
        if original_image_path not in image_mapping:

            missing_images.append(
                original_image_path
            )

            continue

        # NEW FLAT IMAGE NAME
        new_image_name = image_mapping[
            original_image_path
        ]

        # UPDATE ROW
        clean_row = row.copy()
        clean_row["image"] = new_image_name
        valid_rows.append(clean_row)

    except Exception as e:

        print("\nFAILED ROW:", idx)
        print(e)

# CREATE CLEAN DATAFRAME
clean_df = pd.DataFrame(valid_rows)

# SAVE UPDATED CSV
clean_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# DONE
print("VALID ROWS:")
print(len(clean_df))
print("\nMISSING IMAGES:")
print(len(missing_images))

if len(missing_images) > 0:

    print("\nFIRST 20 MISSING:")

    for p in missing_images[:20]:

        print(p)

print("\nOUTPUT IMAGE FOLDER:")
print(OUTPUT_IMAGE_DIR)
print("\nFIXED CSV:")
print(OUTPUT_CSV)

print("\nDONE")