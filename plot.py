import pandas as pd
import matplotlib.pyplot as plt
import sys

# Check if the CSV file path and the column name are provided as command-line arguments
if len(sys.argv) < 3:
    print("Usage: python plot_csv_column.py <csv_file_path> <column_name>")
    sys.exit(1)

# Get the CSV file path and the column name from command-line arguments
csv_file_path = sys.argv[1]
column_name = sys.argv[2]

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)
df.columns = df.columns.str.replace(' ', '')
# Extract the 'epoch' and specified column data
print(df.columns)
epochs = df['epoch']
data = df[f"{column_name}"]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, data, marker='o', linestyle='-')
plt.title(f'{column_name} vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel(column_name)
plt.grid(True)

plt.show()
