import os

# Define the structure
folders = [
    "scripts",
    "venv",
]

files = {
    ".gitignore": """# Ignore virtual environment
venv/

# Python cache files
__pycache__/
*.pyc
*.pyo
*.pyd

# OS specific files
.DS_Store
Thumbs.db

# Logs
*.log
""",
    "requirements.txt": "# Add your Python dependencies here\n",
    "README.md": "# Sentinel Eye Project\n\nA privacy-preserving theft detection system.\n",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Create files
for file_name, content in files.items():
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created file: {file_name}")

print("\nProject structure setup complete!")
