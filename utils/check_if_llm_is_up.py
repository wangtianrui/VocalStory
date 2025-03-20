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

def check_if_llm_is_up(openai_client, model_name):
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Hello, this is a health test. Reply with any word if you're working."}
            ]
        )
        
        return True, response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return False, "Your configured LLM is not working. Please check if the .env file is correctly set up. Error: " + str(e)