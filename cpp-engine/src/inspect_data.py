import pandas as pd

# Load the file
file_path = 'data/raw/FI2010_train.csv'
# Adjust path if you are inside the 'notebooks' folder:
# file_path = '../data/raw/FI2010_train.csv'

print("🔍 INSPECTING RAW DATA...")

# Load first 5 rows
df = pd.read_csv(file_path, nrows=5, header=None)

# 1. Print the shape (How many columns do we actually have?)
print(f"📊 Data Shape: {df.shape}")

# 2. Print the first 5 rows (The raw numbers)
print("\n--- FIRST 5 ROWS (Look for the columns with float numbers) ---")
print(df.head())

# 3. Check specific column values to find the real prices
print("\n--- COLUMN ANALYSIS ---")
print(f"Column 0 (First value): {df.iloc[0, 0]}")
print(f"Column 1 (First value): {df.iloc[0, 1]}")
print(f"Column 2 (First value): {df.iloc[0, 2]}")
print(f"Column 3 (First value): {df.iloc[0, 3]}")