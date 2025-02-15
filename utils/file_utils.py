"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import json
import shutil

def empty_file(file_name):
    # Open the file in write mode to make it empty
    with open(file_name, 'w') as file:
        pass  # No content is written, so the file becomes empty

def empty_directory(directory_path):
    """
    Removes all files and subdirectories within the specified directory.

    Args:
        directory_path (str): The path to the directory to empty.

    This function iterates through each item in the directory. If the item is a file
    or a symbolic link, it is unlinked (deleted). If the item is a directory, it and 
    all its contents are removed using shutil.rmtree.
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            # Check if the current item is a file or symbolic link
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symbolic link
            # Check if the current item is a directory
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            # Handle exceptions and print an error message if deletion fails
            print(f"Failed to delete {file_path}: {e}")

def read_json(filename):
    # Open the JSON file
    with open(filename, 'r') as file:
        # Load the JSON data
        data = json.load(file)

        return data

def write_json_to_file(data, file_name):
    """
    Writes a JSON object to a file.

    Args:
        data (dict): The JSON object to write.
        file_name (str): The name of the file to which to write the JSON object.

    The file is opened in write mode, so the contents of the file will be overwritten.
    If the file does not exist, it will be created.
    """
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def write_jsons_to_jsonl_file(json_objects, file_name):
    """
    Writes a list of JSON objects to a JSONL file.

    Args:
        json_objects (list[dict]): A list of dictionaries representing the JSON objects to write.
        file_name (str): The name of the file to which to write the JSON objects.

    The file is opened in append mode, so the JSON objects will be appended to the
    existing file contents. If the file does not exist, it will be created.
    """
    with open(file_name, 'a') as jsonl_file:
        for obj in json_objects:
            jsonl_file.write(json.dumps(obj) + '\n')