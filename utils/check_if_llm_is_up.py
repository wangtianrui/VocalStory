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