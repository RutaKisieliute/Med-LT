# SPILT CSV INTO TRAIN AND VALIDATE CSV

import pandas as pd

# INPUT CSV
INPUT_CSV = r"input_csv_path"

# OUTPUT CSVs
FIRST_OUTPUT_CSV = r"output_train_csv_path"

SECOND_OUTPUT_CSV = r"output_validate_csv_path"

# NUMBER OF ROWS IN FIRST CSV
ROWS_IN_FIRST_CSV = 16715

# RANDOM SEED
RANDOM_SEED = 42

# LOAD CSV
df = pd.read_csv(INPUT_CSV)

print("\nTOTAL ROWS:")
print(len(df))

# SHUFFLE
df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# SPLIT
df_first = df.iloc[
    :ROWS_IN_FIRST_CSV
]

df_second = df.iloc[
    ROWS_IN_FIRST_CSV:
]

# SAVE
df_first.to_csv(
    FIRST_OUTPUT_CSV,
    index=False
)

df_second.to_csv(
    SECOND_OUTPUT_CSV,
    index=False
)

# DONE
print("\nFIRST CSV:")
print(FIRST_OUTPUT_CSV)

print("ROWS:")
print(len(df_first))

print("\nSECOND CSV:")
print(SECOND_OUTPUT_CSV)

print("ROWS:")
print(len(df_second))

print("\nRANDOM SEED:")
print(RANDOM_SEED)

print("\nDONE")