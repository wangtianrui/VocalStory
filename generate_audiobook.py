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
import time

client = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed"
)

def read_json(filename):
    # Open the JSON file
    with open(filename, 'r') as file:
        # Load the JSON data
        data = json.load(file)

        return data
    
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

def generate_audio_with_single_voice():
    """
    Generates an audiobook using a single voice.

    This function reads text from a file called "converted_book.txt" and generates an
    audiobook using the "af_heart" voice. The speed of the voice is set to 0.85.

    The estimated duration of the audiobook is calculated based on the number of
    words in the text and the words per minute (WPM) of the voice. The estimated
    size of the audiobook is calculated based on the estimated duration and the
    bytes per second of the AAC format.

    The progress of the generation is displayed using a tqdm progress bar.

    The generated audiobook is saved to a file called "generated_audiobooks/audiobook.aac".

    The function prints a message when the generation is complete.
    """
    f = open("converted_book.txt", "r")
    text = f.read()
    num_words = len(text.split())

    # ✅ Updated WPM based on calibration
    words_per_minute = 167.36  # Adjusted WPM from real-world data
    estimated_duration = (num_words / words_per_minute) * 60  # in seconds

    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_heart",
        response_format="aac",  # Ensuring format consistency
        speed=0.85,
        input=text
    ) as response:
        file_path = "generated_audiobooks/audiobook.aac"

        # ✅ Updated size estimate from real-world data
        size_per_second = 16267  # Bytes per second for AAC
        total_size = estimated_duration * size_per_second

        with open(file_path, "wb") as audio_file, tqdm(
            total=int(total_size), unit="B", unit_scale=True, desc="Generating Audio"
        ) as pbar:
            for chunk in response.iter_bytes():
                audio_file.write(chunk)
                pbar.update(len(chunk))

    print("TTS generation complete!")

import json
from tqdm import tqdm

def generate_audio_with_multiple_voices():
    """
    Generates an audiobook with multiple voices by processing a JSONL file containing speaker-attributed lines.

    This function reads a JSONL file where each line represents a JSON object containing a line of text and its
    associated speaker. It maps each speaker to a specific voice based on gender and other criteria, then uses
    a text-to-speech (TTS) service to generate audio for each line. The resulting audio is saved as an AAC file.

    The function also uses a progress bar to track the audio generation process.

    Requirements:
    - A JSONL file named 'speaker_attributed_book.jsonl' containing lines and speaker information.
    - Two JSON files: 'character_gender_map.json' and 'kokoro_voice_map.json' for mapping speakers to voices.
    - A TTS client (e.g., `client.audio.speech`) configured for streaming audio generation.

    Output:
    - An AAC file named 'audiobook.aac' saved in the 'generated_audiobooks' directory.
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
    
    # Get the total number of lines to process for the progress bar
    total_size = len(json_data_array)
    
    # Initialize a progress bar to track the audio generation process
    with tqdm(total=total_size, unit="line", desc="Audio Generation Progress") as overall_pbar:
        # Open an AAC file for writing the generated audio
        with open("generated_audiobooks/audiobook.aac", "wb") as audio_file:
            for doc in json_data_array:
                # Extract the line of text and the speaker from the JSON object
                line = doc["line"]
                speaker = doc["speaker"]
                
                # Find the appropriate voice for the speaker based on gender and voice mapping
                voice = find_voice_for_gender_score(speaker, character_gender_map, kokoro_voice_map)
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Generate audio for the line using the TTS service
                with client.audio.speech.with_streaming_response.create(
                    model="kokoro",
                    voice=voice,
                    response_format="aac",
                    speed=0.85,
                    input=line
                ) as response:
                    # Stream the audio chunks and write them to the AAC file
                    for chunk in response.iter_bytes():
                        audio_file.write(chunk)
                
                # Update the progress bar after processing each line
                overall_pbar.update(1)

    # Print a completion message
    print("TTS generation complete!")

os.makedirs("generated_audiobooks", exist_ok=True)
option = input("Enter 1 for single voice or 2 for multiple voices: ")

start_time = time.time()

if option == "1":
    generate_audio_with_single_voice()
else:
    generate_audio_with_multiple_voices()

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")