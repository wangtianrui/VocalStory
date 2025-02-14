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
    command = f"/usr/bin/ebook-meta '{book_path}' --get-cover cover.jpg"
    
    result = run_shell_command_without_virtualenv(command)

    metadata = {}
    for line in result.stdout.split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            metadata[key.strip()] = value.strip()
    return metadata
    
def get_audio_duration(file_path):
    """Returns the duration of an audio file in milliseconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(float(result.stdout.strip()) * 1000)  # Convert to milliseconds

def generate_chapters_file(chapter_files, output_file="chapters.txt"):
    """Generates a chapter metadata file for FFmpeg."""
    start_time = 0
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for chapter in chapter_files:
            duration = get_audio_duration(os.path.join("temp_audio", chapter))
            end_time = start_time + duration
            
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_time}\n")
            f.write(f"END={end_time}\n")
            f.write(f"title={os.path.splitext(chapter)[0]}\n\n")  # Use filename as chapter title
            
            start_time = end_time  # Update start time for next chapter
    
def merge_chapters_to_m4b(book_path, chapter_files):
    """Uses ffmpeg to merge all chapter files into an M4B audiobook."""
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
    
    output_m4b = "generated_audiobooks/audiobook.m4b"
    cover_image = "cover.jpg"

    # Generate chapter metadata
    generate_chapters_file(chapter_files, "chapters.txt")

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
    """Uses ffmpeg to merge all chapter files into a standard audio file (ex. aac/ mp3)."""
    file_list_path = "chapter_list.txt"
    
    with open(file_list_path, "w") as f:
        for chapter in chapter_files:
            f.write(f"file '{os.path.join('temp_audio', chapter)}'\n")

    output_file = f"generated_audiobooks/audiobook.{output_format}"

    ffmpeg_cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {file_list_path} -c copy {output_file}"
    )
    
    subprocess.run(ffmpeg_cmd, shell=True, check=True)
    print(f"Audiobook created: {output_file}")