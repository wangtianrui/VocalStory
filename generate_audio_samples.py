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
import itertools
import requests
import os

os.makedirs("audio_samples", exist_ok=True)

client = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed"
)

text = """Humpty Dumpty sat on a wall.
Humpty Dumpty had a great fall.
All the king's horses and all the king's men
Couldn't put Humpty together again."""

response = requests.get("http://localhost:8880/v1/audio/voices")
voices_res = response.json()
voices = voices_res["voices"]
print("Available voices:", voices)

combinations = list(itertools.combinations(voices, 2))
all_voices_combinations = []

for comb in combinations:
    all_voices_combinations.append("+".join(comb))

gen_for_all_combinations = input("Generate voice sample for all voice combinations ? Enter yes or no : ")

if(gen_for_all_combinations == "yes"):
    for voice in all_voices_combinations:
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=voice,
            response_format="aac",  # Ensuring format consistency
            speed=0.85,
            input=text
        ) as response:
            file_path = f"audio_samples/{voice}.aac"
            response.stream_to_file(file_path)
else:
    for voice in voices:
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=voice,
            response_format="aac",  # Ensuring format consistency
            speed=0.85,
            input=text
        ) as response:
            file_path = f"audio_samples/{voice}.aac"
            response.stream_to_file(file_path)

print("TTS generation complete!")
