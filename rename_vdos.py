import os

# Path to your shoplifting videos folder
folder = r"D:\sentinel_eye\videos\shoplifting"

# List all mp4 files
files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
files.sort()  # Optional: sort alphabetically

# Rename files sequentially
for i, f in enumerate(files, start=1):
    new_name = f"shoplifting_{i:02d}.mp4"
    os.rename(os.path.join(folder, f), os.path.join(folder, new_name))

print("Renaming completed!")
