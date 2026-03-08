import pandas as pd

# Path to your metrics.csv file
metrics_csv_path = "outputs/metrics.csv"

# Read the CSV into a pandas DataFrame with error handling for bad rows
df = pd.read_csv(metrics_csv_path, on_bad_lines='skip')

# Check if 'latency' column exists
if 'latency' not in df.columns:
    # Add a 'latency' column with default value of 0.0 if it doesn't exist
    df['latency'] = 0.0

# Fill missing 'latency' values with 0.0 (if any)
df['latency'] = df['latency'].fillna(0.0)

# Ensure there are exactly 6 columns (arch_id, generation, val_accuracy, param_count, flops, latency)
df = df[df.columns[:6]]

# Save the cleaned DataFrame back to the CSV file
df.to_csv(metrics_csv_path, index=False)

# Inform the user that cleaning was successful
print("CSV file cleaned and saved successfully.")
