def check_if_kokoro_api_is_up(client):
    try:
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_heart",
            response_format="aac",  # Ensuring format consistency
            speed=0.85,
            input="Hello, how are you ?"
        ) as response:
            return True, None
    except Exception as e:
        return False, "The Kokoro API is not working. Please check if the .env file is correctly set up and the Kokoro API is up. Error: " + str(e)