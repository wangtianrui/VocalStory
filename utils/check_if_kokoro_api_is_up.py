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
import traceback
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

async def check_if_kokoro_api_is_up(client):
    with no_proxy_context():
        try:
            # async with client.audio.speech.with_streaming_response.create(
            #     model="kokoro",
            #     voice="af_heart",
            #     response_format="aac",  # Ensuring format consistency
            #     speed=0.85,
            #     input="Hello, how are you ?"
            # ) as response:
            #     return True, None
            response = client.audio.speech.create(
                model="kokoro",  
                voice="af_bella+af_sky",
                input="Hello world!",
                response_format="mp3"
            )
            return True, None
        except Exception as e:
            traceback.print_exc()
            return False, "The Kokoro API is not working. Please check if the .env file is correctly set up and the Kokoro API is up. Error: " + str(e)