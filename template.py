import os  # Import the os module to interact with the operating system
from pathlib import Path  # Import the Path class from pathlib module for handling file paths
import logging  # Import the logging module for logging messages

# Configure the logging to display messages with time and message content
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of files that we want to create
list_of_files = [
    "src/__init__.py",       # A Python package initialization file
    "src/helper.py",         # A helper module
    "src/prompt.py",         # A prompt module
    ".env",                  # Environment configuration file
    "setup.py",              # Python setup script
    "research/trials.ipynb", # Jupyter notebook for research
    "app.py",                # Main application script
    "store_index.py",        # Index script for storage
    "static/.gitkeep",                # Directory for static files (e.g., CSS, JS)
    "templates/chat.html" ,   # HTML template for chat
    ".gitignore"
    
]

# Iterate over each file path in the list
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert the file path to a Path object
    filedir, filename = os.path.split(filepath)  # Split the path into directory and file name

    # Check if the directory part of the path is not empty
    if filedir:
        os.makedirs(filedir, exist_ok=True)  # Create the directory if it doesn't exist
        logging.info(f"Creating directory: {filedir} for the file {filename}")  # Log directory creation

    # Check if the file doesn't exist or if it exists but is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:  # Open the file in write mode (this creates an empty file)
            pass  # Do nothing else inside the with block
        logging.info(f"Creating empty file: {filepath}")  # Log file creation
    else:
        logging.info(f"{filename} is already created")  # Log that the file already exists
