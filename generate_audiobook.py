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

from openai import OpenAI
from tqdm import tqdm
import json
import os
import re
import time
import sys
from utils.run_shell_commands import check_if_ffmpeg_is_installed, check_if_calibre_is_installed
from utils.file_utils import read_json, empty_directory
from utils.m4b_audiobook_utils import merge_chapters_to_m4b, merge_chapters_to_standard_audio_file
from dotenv import load_dotenv

load_dotenv()

KOKORO_BASE_URL = os.environ.get("KOKORO_BASE_URL")
KOKORO_API_KEY = os.environ.get("KOKORO_API_KEY")

os.makedirs("audio_samples", exist_ok=True)

client = OpenAI(
    base_url=KOKORO_BASE_URL, api_key=KOKORO_API_KEY
)

def split_and_annotate_text(text):
    """Splits text into dialogue and narration while annotating each segment."""
    parts = re.split(r'("[^"]+")', text)  # Keep dialogues in the split result
    annotated_parts = []

    for part in parts:
        if part:  # Ignore empty strings
            annotated_parts.append({
                "text": part,
                "type": "dialogue" if part.startswith('"') and part.endswith('"') else "narration"
            })

    return annotated_parts

def check_and_extract_chapter_heading(line):
    """Detects if a line is a chapter heading based on common patterns."""
    matched_text = re.match(r'^(Chapter|Part)?\s*\d+', line, re.IGNORECASE)

    if bool(matched_text):
        return line, bool(matched_text)
    else:
        return None, False
    
def find_voice_for_gender_score(character: str, character_gender_map, kokoro_voice_map):
    """
    Finds the appropriate voice for a character based on their gender score.

    Args:
        character (str): The name of the character for whom the voice is being determined.
        character_gender_map (dict): A dictionary mapping character names to their gender scores.
        kokoro_voice_map (dict): A dictionary mapping voice identifiers to gender scores.

    Returns:
        str: The voice identifier that matches the character's gender score.
    """

    character_gender_score_doc = character_gender_map["scores"][character.lower()]
    character_gender_score = character_gender_score_doc["gender_score"]
    for voice, score in kokoro_voice_map.items():
        if score == character_gender_score:
            return voice

def generate_audio_with_single_voice(output_format, generate_m4b_audiobook_file=False, book_path=""):
    """
    Generates an audiobook using a single voice for narration and another voice for dialogues.
    Takes in output_format as an argument for the output format of the audio.

    This function reads text from a file called "converted_book.txt" and generates an
    audiobook using the "af_heart" voice as the narrator and "am_fenrir" voice as the dialogue speaker. The speed of the voice is set to 0.85.

    The progress of the generation is displayed using a tqdm progress bar.

    The generated audiobook is saved to a file called "generated_audiobooks/audiobook.{output_format}".

    The function prints a message when the generation is complete.
    """
    f = open("converted_book.txt", "r")
    text = f.read()
    lines = text.split("\n")

    narrator_voice = "af_heart" # voice to be used for narration
    dialogue_voice = "am_fenrir" # voice to be used for dialogue

    # Get the total number of lines to process for the progress bar
    total_size = len(lines)
    chapter_index = 1
    current_chapter_audio = f"Introduction.{output_format}"
    chapter_files = []
    temp_audio_dir = "temp_audio"
    os.makedirs(temp_audio_dir, exist_ok=True)
    empty_directory(temp_audio_dir)

    with tqdm(total=total_size, unit="line", desc="Audio Generation Progress") as overall_pbar:
        # Open a file for writing the generated audio
        with open(f"generated_audiobooks/audiobook.{output_format}", "wb") as combined_audio_file:
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # If the line is a chapter heading, start a new audio file
                match, is_chapter_heading = check_and_extract_chapter_heading(line)
                if is_chapter_heading:
                    chapter_index += 1
                    current_chapter_audio = f"{match}.{output_format}"
                
                chapter_path = os.path.join(temp_audio_dir, current_chapter_audio)

                with open(chapter_path, "ab") as audio_file:  # Append mode
                    annotated_parts = split_and_annotate_text(line) # split the line into annotated parts containing dialogue and narration

                    for part in annotated_parts: # generate audio for each part : either dialogue or narration
                        text_to_speak = part["text"]
                        voice_to_speak_in = narrator_voice
                        if part["type"] == "narration":
                            voice_to_speak_in = narrator_voice
                        elif part["type"] == "dialogue":
                            voice_to_speak_in = dialogue_voice

                        # Generate audio for the line using the TTS service
                        with client.audio.speech.with_streaming_response.create(
                            model="kokoro",
                            voice=voice_to_speak_in,
                            response_format=output_format,
                            speed=0.85,
                            input=text_to_speak
                        ) as response:
                            # Stream the audio chunks and write them to the output file
                            for chunk in response.iter_bytes():
                                audio_file.write(chunk)
                                combined_audio_file.write(chunk)
                                    
                        
                if current_chapter_audio not in chapter_files:
                    chapter_files.append(current_chapter_audio)
                overall_pbar.update(1)

    if(generate_m4b_audiobook_file):
        # Merge all chapter files into a final m4b audiobook
        merge_chapters_to_m4b(book_path, chapter_files)

def generate_audio_with_multiple_voices(output_format, generate_m4b_audiobook_file=False, book_path=""):
    """
    Generates an audiobook with multiple voices by processing a JSONL file containing speaker-attributed lines.
    Takes in output_format as an argument for the output format of the audio.

    This function reads a JSONL file where each line represents a JSON object containing a line of text and its
    associated speaker. It maps each speaker to a specific voice based on gender and other criteria, then uses
    a text-to-speech (TTS) service to generate audio for each line. The resulting audio is saved in the output_format.

    The function also uses a progress bar to track the audio generation process.

    Requirements:
    - A JSONL file named 'speaker_attributed_book.jsonl' containing lines and speaker information.
    - Two JSON files: 'character_gender_map.json' and 'kokoro_voice_map.json' for mapping speakers to voices.
    - A TTS client (e.g., `client.audio.speech`) configured for streaming audio generation.

    Output:
    - An output file named 'audiobook.{output_format}' saved in the 'generated_audiobooks' directory.
    """
    
    # Path to the JSONL file containing speaker-attributed lines
    file_path = 'speaker_attributed_book.jsonl'
    json_data_array = []

    # Open the JSONL file and read it line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            json_object = json.loads(line.strip())
            
            # Append the parsed JSON object to the array
            json_data_array.append(json_object)

    # Load mappings for character gender and voice selection
    character_gender_map = read_json("character_gender_map.json")
    kokoro_voice_map = read_json("kokoro_voice_map.json")
    narrator_voice = find_voice_for_gender_score("narrator", character_gender_map, kokoro_voice_map) # Loading the default narrator voice
    
    # Get the total number of lines to process for the progress bar
    total_size = len(json_data_array)
    chapter_index = 1
    current_chapter_audio = f"Introduction.{output_format}"
    chapter_files = []
    temp_audio_dir = "temp_audio"
    os.makedirs(temp_audio_dir, exist_ok=True)
    empty_directory(temp_audio_dir)
    
    # Initialize a progress bar to track the audio generation process
    with tqdm(total=total_size, unit="line", desc="Audio Generation Progress") as overall_pbar:
        # Open a file for writing the generated audio
        with open(f"generated_audiobooks/audiobook.{output_format}", "wb") as combined_audio_file:
            for doc in json_data_array:
                # Extract the line of text and the speaker from the JSON object
                line = doc["line"].strip()

                # Skip empty lines
                if not line:
                    continue

                speaker = doc["speaker"]
                
                # Find the appropriate voice for the speaker based on gender and voice mapping
                speaker_voice = find_voice_for_gender_score(speaker, character_gender_map, kokoro_voice_map)

                # If the line is a chapter heading, start a new audio file
                match, is_chapter_heading = check_and_extract_chapter_heading(line)
                if is_chapter_heading:
                    chapter_index += 1
                    current_chapter_audio = f"{match}.{output_format}"
                
                chapter_path = os.path.join(temp_audio_dir, current_chapter_audio)

                with open(chapter_path, "ab") as audio_file:  # Append mode
                    annotated_parts = split_and_annotate_text(line) # split the line into annotated parts containing dialogue and narration

                    for part in annotated_parts: # generate audio for each part : either dialogue or narration
                        text_to_speak = part["text"]
                        voice_to_speak_in = narrator_voice
                        if part["type"] == "narration":
                            voice_to_speak_in = narrator_voice
                        elif part["type"] == "dialogue":
                            voice_to_speak_in = speaker_voice

                        # Generate audio for the line using the TTS service
                        with client.audio.speech.with_streaming_response.create(
                            model="kokoro",
                            voice=voice_to_speak_in,
                            response_format=output_format,
                            speed=0.85,
                            input=text_to_speak
                        ) as response:
                            # Stream the audio chunks and write them to the output file
                            for chunk in response.iter_bytes():
                                audio_file.write(chunk)
                                combined_audio_file.write(chunk)
                    
                if current_chapter_audio not in chapter_files:
                    chapter_files.append(current_chapter_audio)
                overall_pbar.update(1)

    if(generate_m4b_audiobook_file):
        # Merge all chapter files into a final m4b audiobook
        merge_chapters_to_m4b(book_path, chapter_files)

def main():
    os.makedirs("generated_audiobooks", exist_ok=True)

    # Default values
    book_path = "./sample_book_and_audio/Adventure of the Lost Treasure, The - Prakhar Sharma.epub"
    generate_m4b_audiobook_file = False
    output_format = "aac"

    # Prompt user for voice selection
    print("\n🎙️ **Audiobook Voice Selection**")
    voice_option = input("🔹 Enter **1** for **Single Voice** or **2** for **Multiple Voices**: ").strip()

    # Prompt user for audiobook type selection
    print("\n🎙️ **Audiobook Type Selection**")
    print("🔹 Do you want the audiobook in M4B format (the standard format for audiobooks) with chapter timestamps and embedded book cover ? (Needs calibre and ffmpeg installed)")
    print("🔹 OR do you want a standard audio file in an AAC/ MP3 format without any of the above features ?")
    audiobook_type_option = input("🔹 Enter **1** for **M4B audiobook format** or **2** for **Standard Audio File (AAC/MP3)**: ").strip()

    if audiobook_type_option == "1":
        is_calibre_installed = check_if_calibre_is_installed()

        if not is_calibre_installed:
            print("⚠️ Calibre is not installed. Please install it first and make sure **calibre** and **ebook-meta** commands are available in your PATH.")
            return
        
        is_ffmpeg_installed = check_if_ffmpeg_is_installed()

        if not is_ffmpeg_installed:
            print("⚠️ FFMpeg is not installed. Please install it first and make sure **ffmpeg** and **ffprobe** commands are available in your PATH.")
            return

        # Check if a path is provided via command-line arguments
        if len(sys.argv) > 1:
            book_path = sys.argv[1]
            print(f"📂 Using book file from command-line argument: **{book_path}**")
        else:
            # Ask user for book file path if not provided
            input_path = input("\n📖 Enter the **path to the book file**, needed for metadata and cover extraction. (Press Enter to use default): ").strip()
            if input_path:
                book_path = input_path
            print(f"📂 Using book file: **{book_path}**")

        print("✅ Book path set. Proceeding...\n")

        generate_m4b_audiobook_file = True
    else:
        # Prompt user for audio format selection
        print("\n🎙️ **Audiobook Output Format Selection**")
        output_format = input("🔹 Choose between ['aac', 'mp3']. Other formats ['opus', 'flac', 'wav', 'pcm'] give incomplete audio or have error in them in Kokoro : ").strip()

        if(output_format not in ['aac', 'mp3']):
            print("\n⚠️ Invalid output format! Please choose either 'aac' or 'mp3'.")
            return

    start_time = time.time()

    if voice_option == "1":
        print("\n🎧 Generating audiobook with a **single voice**...")
        generate_audio_with_single_voice(output_format, generate_m4b_audiobook_file, book_path)
    elif voice_option == "2":
        print("\n🎭 Generating audiobook with **multiple voices**...")
        generate_audio_with_multiple_voices(output_format, generate_m4b_audiobook_file, book_path)
    else:
        print("\n⚠️ Invalid option! Please restart and enter either **1** or **2**.")
        return

    print(f"\n🎧 Audiobook is generated ! The audiobook is saved as **audiobook.{output_format}** in the **generated_audiobooks** directory in the current folder.")

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"\n⏱️ **Execution Time:** {execution_time:.6f} seconds\n✅ Audiobook generation complete!")

if __name__ == "__main__":
    main()