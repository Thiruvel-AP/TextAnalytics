import os
import shutil

def store_dataset(source_path, target_directory, target_filename):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Directory '{target_directory}' created.")

    destination = os.path.join(target_directory, target_filename)

    try:
        # Move the file
        shutil.move(source_path, destination)
        print(f"Successfully stored: {destination}")
    except FileNotFoundError:
        print(f"Error: No file found at {source_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")