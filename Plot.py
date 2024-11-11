import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set a seaborn style for better aesthetics
sns.set(style='whitegrid')

# Read the CSV file
data = pd.read_csv('Results.csv')
# Extract the relevant columns
operating_power = data['Operating Power (dBm)']
original_efficiency = data['Mean Original Edge Rate']
final_efficiency = data['Mean Final Edge Rate']

# Plot the data with enhancements
plt.figure(figsize=(12, 8))
plt.plot(operating_power, original_efficiency, label='Mean Original Edge Rate', marker='o', linestyle='-', color='royalblue', markersize=8)
plt.plot(operating_power, final_efficiency, label='Mean Final Edge Rate', marker='x', linestyle='--', color='orange', markersize=8)
# Add labels and title with larger fonts
plt.xlabel('Operating Power (dBm)', fontsize=14)
plt.ylabel('Edge Rates', fontsize=14)
plt.title('Edge Rates vs Operating Power', fontsize=16, fontweight='bold')

# Enhance the legend
plt.legend(fontsize=12, loc='best')

# Add a grid with custom settings
plt.grid(True, which='both', linestyle='--', linewidth=0.6)

# Show the plot with tight layout
plt.tight_layout()
plt.show()
