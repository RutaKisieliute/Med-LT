# SPLIT CSV INTO TRAIN VALIDATE AND TEST CSV

import os
import pandas as pd

# INPUT CSV
INPUT_CSV = r"input_csv_path"

# OUTPUT DIRECTORY
OUTPUT_DIR = r"output_folder_path"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# SPLIT ACCORDING TO START OF IMAGE NAME IN CSV
SPLIT_RULES = {

    "test": "slake-test.csv",
    "train": "slake-train.csv",
    "validate": "slake-validate.csv",
}

# LOAD CSV
df = pd.read_csv(INPUT_CSV)
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# VERIFY IMAGE COLUMN EXIST
if "image" not in df.columns:

    raise ValueError(
        "CSV must contain image column"
    )

# STORAGE
split_data = {}

for output_name in SPLIT_RULES.values():

    split_data[output_name] = []

unmatched_rows = []

# PROCESS ROWS
for idx, row in df.iterrows():

    image_path = str(
        row["image"]
    ).replace(
        "\\",
        "/"
    )

    matched = False

    # CHECK RULES
    for prefix, output_csv in SPLIT_RULES.items():

        if image_path.startswith(prefix):

            split_data[
                output_csv
            ].append(row)

            matched = True

            break

    # NO MATCH
    if not matched:

        unmatched_rows.append(row)

# SAVE SPLIT CSV
for output_csv, rows in split_data.items():

    output_path = os.path.join(
        OUTPUT_DIR,
        output_csv
    )

    output_df = pd.DataFrame(rows)

    output_df.to_csv(
        output_path,
        index=False
    )

    print("\nSAVED:")
    print(output_path)

    print("ROWS:")
    print(len(output_df))

# SAVE UNMATCHED IF THERE ARE ANY
if len(unmatched_rows) > 0:

    unmatched_path = os.path.join(
        OUTPUT_DIR,
        "unmatched.csv"
    )

    unmatched_df = pd.DataFrame(
        unmatched_rows
    )

    unmatched_df.to_csv(
        unmatched_path,
        index=False
    )

    print("\nUNMATCHED ROWS:")
    print(len(unmatched_df))

# DONE
print("DONE")