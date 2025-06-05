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
from tqdm import tqdm
from dotenv import load_dotenv
import os
from contextlib import contextmanager

@contextmanager
def no_proxy_context():
    # 保存旧代理设置
    old_http_proxy = os.environ.get("http_proxy")
    old_https_proxy = os.environ.get("https_proxy")
    try:
        # 暂时关闭代理
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        yield
    finally:
        # 恢复原代理
        if old_http_proxy is not None:
            os.environ["http_proxy"] = old_http_proxy
        else:
            os.environ.pop("http_proxy", None)
        if old_https_proxy is not None:
            os.environ["https_proxy"] = old_https_proxy
        else:
            os.environ.pop("https_proxy", None)
            
load_dotenv()

KOKORO_BASE_URL = os.environ.get("KOKORO_BASE_URL", "http://localhost:8880/v1")
KOKORO_API_KEY = os.environ.get("KOKORO_API_KEY", "not-needed")

os.makedirs("audio_samples", exist_ok=True)

from httpx import AsyncHTTPTransport
from httpx import HTTPTransport
import httpx
transport = HTTPTransport(proxy=None)
http_client = httpx.Client(transport=transport)
client = OpenAI(
    base_url=KOKORO_BASE_URL, api_key=KOKORO_API_KEY,
    http_client=http_client
)

text = """Humpty Dumpty sat on a wall.
Humpty Dumpty had a great fall.
All the king's horses and all the king's men
Couldn't put Humpty together again."""

response = requests.get(f"{KOKORO_BASE_URL}/audio/voices")
voices_res = response.json()
voices = voices_res["voices"]
# print("Available voices:", voices)

combinations = list(itertools.combinations(voices, 2))
all_voices_combinations = voices.copy()

for comb in combinations:
    all_voices_combinations.append("+".join(comb))

gen_for_all_combinations = input("Generate voice sample for all voice combinations ? Enter yes or no : ")
gen_for_all_combinations = gen_for_all_combinations.strip().lower()

with no_proxy_context():
    if(gen_for_all_combinations == "yes"):
        with tqdm(total=len(all_voices_combinations), unit="line", desc="Audio Generation Progress") as overall_pbar:
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
                overall_pbar.update(1)
    else:
        with tqdm(total=len(voices), unit="line", desc="Audio Generation Progress") as overall_pbar:
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
                overall_pbar.update(1)

    print("TTS generation complete!")
