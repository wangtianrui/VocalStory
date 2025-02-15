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

import subprocess
import os
from utils.run_shell_commands import run_shell_command_without_virtualenv

# Escape double quotes by replacing them with \"
def escape_metadata(value):
    if value:
        return value.replace('"', '\\"')  # Escape double quotes
    return ""

def get_ebook_metadata_with_cover(book_path):
    """
    Extracts metadata from an ebook and saves its cover image.

    Args:
        book_path (str): The path to the ebook file.

    Returns:
        dict: A dictionary containing the ebook's metadata.
    """
    # Command to extract metadata and cover image using ebook-meta
    command = f"/usr/bin/ebook-meta '{book_path}' --get-cover cover.jpg"

    # Run the command and capture the result
    result = run_shell_command_without_virtualenv(command)

    metadata = {}
    # Parse the command output to extract metadata
    for line in result.stdout.split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            metadata[key.strip()] = value.strip()
    
    return metadata
    
def get_audio_duration(file_path):
    """
    Returns the duration of an audio file in milliseconds using ffprobe.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        int: The duration of the audio file in milliseconds.
    """
    # Construct the command to execute
    cmd = [
        "ffprobe",  # Use ffprobe to get the duration
        "-v", "error",  # Set the verbosity to error
        "-show_entries",  # Show the specified entries
        "format=duration",  # Show the duration
        "-of",  # Specify the output format
        "default=noprint_wrappers=1:nokey=1",  # Print the duration without any additional information
        file_path  # Specify the file to analyze
    ]
    # Run the command and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Convert the output to an integer (in milliseconds) and return it
    return int(float(result.stdout.strip()) * 1000)

def generate_chapters_file(chapter_files, output_file="chapters.txt"):
    """
    Generates a chapter metadata file for FFmpeg.

    The chapter metadata file is a text file that contains information about each chapter in the audiobook, such as the chapter title and the start and end times of the chapter.

    Args:
        chapter_files (list): A list of the paths to the individual chapter audio files.
        output_file (str): The path to the output chapter metadata file. Defaults to "chapters.txt".
    """
    start_time = 0
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for chapter in chapter_files:
            duration = get_audio_duration(os.path.join("temp_audio", chapter))
            end_time = start_time + duration
            
            # Write the chapter metadata to the file
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_time}\n")
            f.write(f"END={end_time}\n")
            f.write(f"title={os.path.splitext(chapter)[0]}\n\n")  # Use filename as chapter title
            
            # Update the start time for the next chapter
            start_time = end_time
    
def merge_chapters_to_m4b(book_path, chapter_files):
    """
    Uses ffmpeg to merge all chapter files into an M4B audiobook.

    This function takes the path to the book file and a list of chapter files as input, and generates an M4B audiobook with chapter metadata and a cover image.

    Args:
        book_path (str): The path to the book file.
        chapter_files (list): A list of the paths to the individual chapter audio files.
    """
    file_list_path = "chapter_list.txt"
    
    with open(file_list_path, "w") as f:
        for chapter in chapter_files:
            f.write(f"file '{os.path.join('temp_audio', chapter)}'\n")

    metadata = get_ebook_metadata_with_cover(book_path)
    title = escape_metadata(metadata.get("Title", ""))
    authors = escape_metadata(metadata.get("Author(s)", ""))
    publisher = escape_metadata(metadata.get("Publisher", ""))
    languages = escape_metadata(metadata.get("Languages", ""))
    published_date = escape_metadata(metadata.get("Published", ""))
    comments = escape_metadata(metadata.get("Comments", ""))
    
    # Generate chapter metadata
    generate_chapters_file(chapter_files, "chapters.txt")

    output_m4b = "generated_audiobooks/audiobook.m4b"
    cover_image = "cover.jpg"

    # Construct metadata arguments safely
    metadata = (
        f"-metadata title=\"{title}\" "
        f"-metadata artist=\"{authors}\" "
        f"-metadata album=\"{title}\" "
        f"-metadata genre=\"Audiobook\" "
        f"-metadata publisher=\"{publisher}\" "
        f"-metadata language=\"{languages}\" "
        f"-metadata date=\"{published_date}\" "
        f"-metadata description=\"{comments}\""
    )
    
    ffmpeg_cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {file_list_path} -i {cover_image} -i chapters.txt "
        f"-c copy -map 0 -map 1 -disposition:v:0 attached_pic -map_metadata 2 {metadata} {output_m4b}"
    )
    
    subprocess.run(ffmpeg_cmd, shell=True, check=True)
    print(f"Audiobook created: {output_m4b}")

def merge_chapters_to_standard_audio_file(chapter_files, output_format):
    """
    Uses ffmpeg to merge all chapter files into a standard audio file (ex. aac/ mp3).

    This function takes a list of chapter files and an output format as input, and generates a standard audio file with the specified format.

    Args:
        chapter_files (list): A list of the paths to the individual chapter audio files.
        output_format (str): The desired output format for the audio file (e.g. aac, mp3).
    """
    file_list_path = "chapter_list.txt"
    
    # Write the list of chapter files to a text file (ffmpeg input)
    with open(file_list_path, "w") as f:
        for chapter in chapter_files:
            f.write(f"file '{os.path.join('temp_audio', chapter)}'\n")

    # Construct the output file path
    output_file = f"generated_audiobooks/audiobook.{output_format}"

    # Construct the ffmpeg command
    ffmpeg_cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {file_list_path} -c copy {output_file}"
    )

    # Run the ffmpeg command
    subprocess.run(ffmpeg_cmd, shell=True, check=True)

    # Print a message when the generation is complete
    print(f"Audiobook created: {output_file}")
