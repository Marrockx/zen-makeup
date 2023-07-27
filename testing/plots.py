from convert_csv import read_csv_and_exclude_first_row
import matplotlib.pyplot as plt

# Replace 'file_path' with the path to your CSV file
file_path = 'data-wo.csv'
data = read_csv_and_exclude_first_row(file_path)


# Unpack the data
successful_frames, total_frames, fps, success_rate = zip(*data[-25:])

# Create two subplots for line plot and bar plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Line plot - Successful Frames vs Total Frames
ax1.plot(total_frames, successful_frames, marker='o', label='Successful Frames')
ax1.set_ylabel('Successful Frames')
ax1.set_title('Successful Frames vs Total Frames')
ax1.legend()

# Bar plot - FPS and Success Rate
plt.scatter(total_frames, fps, s=50, alpha=0.5, c='b', marker='o', label='Data Points')
plt.xlabel('Total frames')
plt.ylabel('Frames per second')
plt.title('Scatter Plot of Data')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()